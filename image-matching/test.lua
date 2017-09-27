require 'cutorch';
require 'cunn';
require 'cudnn'
require 'image'
require 'utils'
require 'DataLoader'
require 'hdf5'

-- User options (change me)
cutorch.setDevice(1)
local snapshotsFolder = 'snapshots-with-class'
local trainMode = 1 -- the training mode set during training
local doCrop = false;
local doFlip = false;

-- Create data sampler
local prodImgsPathFile = 'data/test-product-imgs.txt'
local prodLabelsFile = 'data/test-product-labels.txt'
local testImgsPathFile = 'data/test-imgs.txt'
local testLabelsFile = 'data/test-labels.txt'
local dataLoader = DataLoader(prodImgsPathFile,prodLabelsFile,testImgsPathFile,testLabelsFile,doCrop,doFlip,trainMode)

-- Loop through each snapshot and compute test image features
for snapshotIdx = 17,17 do
    local snapshotName = 'snapshot-'..snapshotIdx..'0000'
    print(snapshotName)

    -- Load trained model
    local modelFile = paths.concat(snapshotsFolder,snapshotName..'.t7');
    local model = torch.load(modelFile)
    local imgEncoder = model:get(1):get(1):clone()
    imgEncoder:evaluate()
    local prodEncoder = model:get(1):get(2):clone()
    if trainMode == 4 or trainMode == 5 then
        prodEncoder:evaluate()
    end

    -- Save product image features
    local prodImgFeats = torch.zeros(dataLoader.numProdImgs,2048)
    local prodImgIdx = 1
    for objIdx = 1,dataLoader.numProdClasses do
        for imgIdx = 1,#dataLoader.prodImgPaths[objIdx] do
            if trainMode == 1 or trainMode == 2 or trainMode == 3 then
                prodImgFeats[prodImgIdx] = dataLoader.prodImgFeats[objIdx][imgIdx]
            else
                print('Generating product image: '..imgIdx..' '..objIdx..'/'..dataLoader.numProdClasses)
                local prodImg = image.load(dataLoader.prodImgPaths[objIdx][imgIdx])
                prodImg = image.scale(prodImg,224,224)
                local mean = {0.485,0.456,0.406}
                local std = {0.229,0.224,0.225}
                for c=1,3 do
                    prodImg[c]:add(-mean[c])
                    prodImg[c]:div(std[c])
                end
                prodImgFeats[prodImgIdx] = prodEncoder:forward(prodImg:cuda()):float()

            end
            prodImgIdx = prodImgIdx+1
        end
    end

    local resultsFile = hdf5.open(paths.concat(snapshotsFolder,'results-'..snapshotName..'.h5'), 'w')
    resultsFile:write('prodFeat', prodImgFeats:float())

    -- Load test image
    local testImgPaths = getLinesFromFile(testImgsPathFile)
    local testImgFeats = torch.ones(#testImgPaths,2048)
    for imgIdx = 1,#testImgPaths do
        print('Testing: '..imgIdx..'/'..#testImgPaths)
        local testImg = image.load(testImgPaths[imgIdx])
        testImg = image.crop(testImg,120,30,520,430)
        testImg = image.scale(testImg,224,224)
        local mean = {0.485,0.456,0.406}
        local std = {0.229,0.224,0.225}
        for c=1,3 do
            testImg[c]:add(-mean[c])
            testImg[c]:div(std[c])
        end

        -- Compute forward pass
        testImgFeats[imgIdx] = imgEncoder:forward(testImg:cuda()):float()
    end

    -- Save test image features
    resultsFile:write('testFeat', testImgFeats:float())
    resultsFile:close()

end
print('Finished.')



