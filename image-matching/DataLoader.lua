require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

local DataLoader = torch.class('DataLoader')

-- Initialize multi-thread data loader
function DataLoader:__init(prodImgPathFile,prodLabelFile,trainImgPathFile,trainLabelFile,doCrop,doFlip,trainMode)

    self.doCrop = doCrop
    self.doFlip = doFlip
    self.trainMode = trainMode

    -- Read paths for training images
    self.trainImgPaths = getLinesFromFile(trainImgPathFile)
    self.trainLabels = getLinesFromFile(trainLabelFile)
    self.numTrainImgs = #self.trainImgPaths

    -- Initialize shuffle indices and count number of classes
    self.shuffleIdx = {}
    self.numTrainClasses = 0;
    for imgIdx = 1,self.numTrainImgs do
        self.trainLabels[imgIdx] = tonumber(self.trainLabels[imgIdx])
        self.shuffleIdx[imgIdx] = imgIdx
        self.numTrainClasses = math.max(self.numTrainClasses,self.trainLabels[imgIdx])
    end

    -- print(self.trainImgPaths)
    -- print(self.trainLabels)
    -- print(self.numTrainClasses)

    -- Shuffle training data
    self.shuffleIdx = shuffleTable(self.shuffleIdx,self.numTrainImgs)
    self.trainIdx = 1
    self.trainEpochIdx = 1
    self.trainEpochSize = self.numTrainImgs

    -- print(self.shuffleIdx)

    -- Read paths for product images
    self.prodImgPathsRaw = getLinesFromFile(prodImgPathFile)
    self.prodLabels = getLinesFromFile(prodLabelFile)
    self.numProdImgs = #self.prodImgPathsRaw

    -- print(self.prodImgPathsRaw)
    -- print(self.prodLabels)
    -- print(self.numProdImgs)

    -- Split product image path table by object class
    self.prodImgPaths = {}
    self.numProdClasses = 0;
    for imgIdx = 1,self.numProdImgs do
        self.prodLabels[imgIdx] = tonumber(self.prodLabels[imgIdx])
        self.numProdClasses = math.max(self.numProdClasses,self.prodLabels[imgIdx])
        if self.prodImgPaths[self.prodLabels[imgIdx]] == nil then
            self.prodImgPaths[self.prodLabels[imgIdx]] = {}
            self.prodImgPaths[self.prodLabels[imgIdx]][1] = self.prodImgPathsRaw[imgIdx]
        else
            self.prodImgPaths[self.prodLabels[imgIdx]][#self.prodImgPaths[self.prodLabels[imgIdx]]+1] = self.prodImgPathsRaw[imgIdx]
        end
    end

    -- print(self.prodImgPaths)
    -- print(self.numProdClasses)

    -- -- Read object types
    -- self.prodObjTypes = getLinesFromFile(prodObjTypesFile)
    -- self.numObjTypes = 7;
    -- for objIdx = 1,self.numProdClasses do
    --     self.prodObjTypes[objIdx] = tonumber(self.prodObjTypes[objIdx])
    -- end

    -- print(self.prodObjTypes)

    if self.trainMode == 1 or self.trainMode == 2 or self.trainMode == 3 then
        -- Pre-compute features for product images
        local model = torch.load('resnet-50.t7')
        model:remove(11)
        model:insert(nn.Normalize(2))
        model = model:cuda()
        model:evaluate()
        self.prodImgFeats = {}
        for objIdx = 1,self.numProdClasses do
            print('Computing product image features: '..objIdx..'/'..self.numProdClasses)
            self.prodImgFeats[objIdx] = {}
            for imgIdx = 1,#self.prodImgPaths[objIdx] do

                -- Load and pre-process product image
                local img = image.load(self.prodImgPaths[objIdx][imgIdx])
                img = image.scale(img,224,224)
                local mean = {0.485,0.456,0.406}
                local std = {0.229,0.224,0.225}
                for c=1,3 do
                    img[c]:add(-mean[c])
                    img[c]:div(std[c])
                end

                -- Get features with a forward pass
                self.prodImgFeats[objIdx][imgIdx] = model:forward(img:cuda()):float()[1]
            end
        end
    end

    -- Define multi-thread pool
    self.batchSize = 6
    self.nthread = self.batchSize
    self.pool = threads.Threads(self.nthread,
        function(threadid)
            pcall(require, 'image')
            math.randomseed(threadid)
        end)
end

-- Load training batch with multi-thread
function DataLoader:getTrainingBatch()
    local input = {torch.zeros(self.batchSize,3,224,224),torch.zeros(self.batchSize,2048),torch.zeros(self.batchSize,3,224,224)}
    if self.trainMode == 4 or self.trainMode == 5 then
        input = {torch.zeros(self.batchSize,3,224,224),torch.zeros(self.batchSize,3,224,224),torch.zeros(self.batchSize,3,224,224)}
    end
    local label = torch.zeros(self.batchSize,1)

    -- Get paths of training images to be loaded
    local batchImgPaths = {}
    local batchImgLabels = {}
    for sampleIdx = 1,self.batchSize do
        batchImgPaths[sampleIdx] = {}
        batchImgPaths[sampleIdx][1] = self.trainImgPaths[self.shuffleIdx[self.trainIdx]]
        local anchorLabel = self.trainLabels[self.shuffleIdx[self.trainIdx]]
        label[sampleIdx][1] = anchorLabel

        -- Pick an image of a random different object
        local randObjIdx = anchorLabel
        while randObjIdx == anchorLabel do
            local randImgIdx = math.floor(math.random()*self.numTrainImgs)+1
            batchImgPaths[sampleIdx][2] = self.trainImgPaths[randImgIdx]
            randObjIdx = self.trainLabels[randImgIdx]
        end

        if self.trainMode == 4 or self.trainMode == 5 then
            local randImgIdx = math.floor(math.random()*#self.prodImgPaths[anchorLabel])+1
            batchImgPaths[sampleIdx][3] = self.prodImgPaths[anchorLabel][randImgIdx]
        else
            -- Set default anchor feature (first product image)
            input[2][sampleIdx] = self.prodImgFeats[anchorLabel][1]
        end

        -- Re-shuffle data if at end of training epoch
        if self.trainIdx == self.trainEpochSize then
            self.shuffleIdx = shuffleTable(self.shuffleIdx,self.numTrainImgs)
            self.trainIdx = 1
            self.trainEpochIdx = self.trainEpochIdx+1
        else
            self.trainIdx = self.trainIdx+1
        end
    end

    -- print(batchImgPaths)

    local doCrop = self.doCrop
    local doFlip = self.doFlip
    local trainMode = self.trainMode

    for jobIdx = 1,self.nthread do
        self.pool:addjob(
            function()

                -- Load and pre-process match image
                local img = image.load(batchImgPaths[jobIdx][1])
                if doCrop then
                    -- img = image.crop(img,120,40,520,440)
                    img = image.crop(img,120,30,520,430)
                end

                if doFlip and torch.uniform()>0.7 then
                   img = image.hflip(img)
                end
                img = image.scale(img,224,224)
                local mean = {0.485,0.456,0.406}
                local std = {0.229,0.224,0.225}
                for c=1,3 do
                    img[c]:add(-mean[c])
                    img[c]:div(std[c])
                end
                input[1][jobIdx] = img:reshape(1,3,224,224);

                if trainMode == 4 or trainMode == 5 then
                    -- Load and pre-process product image
                    local img = image.load(batchImgPaths[jobIdx][3])
                    img = image.scale(img,224,224)
                    local mean = {0.485,0.456,0.406}
                    local std = {0.229,0.224,0.225}
                    for c=1,3 do
                        img[c]:add(-mean[c])
                        img[c]:div(std[c])
                    end
                    input[2][jobIdx] = img:reshape(1,3,224,224);
                end

                -- Load and pre-process non-matching image
                local img = image.load(batchImgPaths[jobIdx][2])
                if doCrop then
                    -- img = image.crop(img,120,40,520,440)
                    img = image.crop(img,120,30,520,430)
                end
                if doFlip and torch.uniform()>0.7 then
                   img = image.hflip(img)
                end
                img = image.scale(img,224,224)
                local mean = {0.485,0.456,0.406}
                local std = {0.229,0.224,0.225}
                for c=1,3 do
                    img[c]:add(-mean[c])
                    img[c]:div(std[c])
                end
                input[3][jobIdx] = img:reshape(1,3,224,224);

                return __threadid
            end)
    end

    self.pool:synchronize()
    collectgarbage()

    return input,label
end

-- -- Load and pre-process depth image
-- local depth = image.load(batchImgPaths[5*(jobIdx-1)+i])
-- depth = depth*65536/1000
-- depth = depth:clamp(0.0,1.0) -- Depth range of Intel RealSense F200
-- depth = image.scale(depth,224,224,'simple')
-- local img = depth:cat(depth,1):cat(depth,1)
-- local mean = {0.485,0.456,0.406}
-- local std = {0.229,0.224,0.225}
-- for c=1,3 do
--     img[c]:add(-mean[c])
--     img[c]:div(std[c])
-- end
