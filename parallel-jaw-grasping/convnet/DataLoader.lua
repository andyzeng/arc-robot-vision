require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'util.lua'

local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

local DataLoader = torch.class('DataLoader')

-- Initialize multi-thread data loader
function DataLoader:__init(options)

    -- Load user options
    self.dataPath = options.dataPath
    self.shuffleData = options.shuffleData
    self.outputScale = options.outputScale
    self.imgHeight = options.imgHeight
    self.imgWidth = options.imgWidth

    -- Read paths for data samples
    self.samplePaths = getLinesFromFile(options.sampleList)
    self.numSamples = #self.samplePaths

    -- Initialize shuffle indices
    self.shuffleIdx = {}
    for imgIdx = 1,self.numSamples do
        self.shuffleIdx[imgIdx] = imgIdx
    end

    -- Shuffle data samples
    if self.shuffleData then
        print('Shuffling training data')
        self.shuffleIdx = shuffleTable(self.shuffleIdx,self.numSamples)
    end

    self.trainIdx = 1
    self.trainEpochIdx = 1
    self.trainEpochSize = self.numSamples

    -- Define multi-thread pool
    self.batchSize = options.batchSize
    self.nthread = self.batchSize
    self.pool = threads.Threads(self.nthread,
        function(threadid)
            pcall(require, 'image')
            math.randomseed(threadid)
        end)
end

-- Load mini-batch with multi-thread
function DataLoader:getMiniBatch()
    local input = {torch.zeros(self.batchSize,3,self.imgHeight,self.imgWidth),torch.zeros(self.batchSize,3,self.imgHeight,self.imgWidth)}
    local label = torch.zeros(self.batchSize,self.imgHeight/self.outputScale,self.imgWidth/self.outputScale)

    -- Get file paths of data samples to be loaded
    local batchColorPaths = {}
    local batchDepthPaths = {}
    local batchLabelPaths = {}
    for sampleIdx = 1,self.batchSize do
        batchColorPaths[sampleIdx] = paths.concat(self.dataPath,'color',self.samplePaths[self.shuffleIdx[self.trainIdx]]..'.png')
        batchDepthPaths[sampleIdx] = paths.concat(self.dataPath,'depth',self.samplePaths[self.shuffleIdx[self.trainIdx]]..'.png')
        batchLabelPaths[sampleIdx] = paths.concat(self.dataPath,'label-aug',self.samplePaths[self.shuffleIdx[self.trainIdx]]..'.png')

        -- Re-shuffle data if at end of training epoch
        if self.trainIdx == self.trainEpochSize then
            if self.shuffleData then
               print('Shuffling training data...')
               self.shuffleIdx = shuffleTable(self.shuffleIdx,self.numSamples)
            end
            self.trainIdx = 1
            self.trainEpochIdx = self.trainEpochIdx+1
        else
            self.trainIdx = self.trainIdx+1
        end
    end

    -- Define dataset average constants (r,g,b)
    local mean = {0.485,0.456,0.406}
    local std = {0.229,0.224,0.225}

    for jobIdx = 1,self.nthread do

        -- Load and pre-process color image
        local colorImg = image.load(batchColorPaths[jobIdx])
        for c=1,3 do
            colorImg[c]:add(-mean[c])
            colorImg[c]:div(std[c])
        end

        -- Load and pre-process depth image
        local depth = image.load(batchDepthPaths[jobIdx])
        depth = depth*65536/10000
        depth = depth:clamp(0.0,1.2) -- Depth range of Intel RealSense SR300
        local depthImg = depth:cat(depth,1):cat(depth,1)
        for c=1,3 do
            depthImg[c]:add(-mean[c])
            depthImg[c]:div(std[c])
        end

        -- Load and pre-process labels
        local labelImg = torch.round(image.load(batchLabelPaths[jobIdx])*2)+1

        input[1][jobIdx]:copy(colorImg:reshape(1,3,self.imgHeight,self.imgWidth))
        input[2][jobIdx]:copy(depthImg:reshape(1,3,self.imgHeight,self.imgWidth))

	    labelImg = image.scale(labelImg, self.imgWidth/self.outputScale,self.imgHeight/self.outputScale,'simple')
        label[jobIdx]:copy(labelImg)
    end

    collectgarbage()

    return input,label
end
