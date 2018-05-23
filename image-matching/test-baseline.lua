require 'cutorch';
require 'cunn';
require 'cudnn'
require 'image'

require 'utils.lua'
require 'DataLoader.lua'

require 'hdf5'

cutorch.setDevice(1)

local snapshotsFolder = '.'
local snapshotName = 'resnet-50'

local trainMode = 1

-- Load trained model
local modelFile = paths.concat(snapshotsFolder,snapshotName..'.t7');
local model = torch.load(modelFile)
model:remove(11)
model:insert(nn.Normalize(2))
model = model:cuda()
local imgEncoder = model:clone()
imgEncoder:evaluate()

local doCrop = false;
local doFlip = false;

-- Create data sampler
local prodImgsPathFile = 'data/test-product-imgs.txt'
local prodLabelsFile = 'data/test-product-labels.txt'
local testImgsPathFile = 'data/test-imgs.txt'
local testLabelsFile = 'data/test-labels.txt'
local dataLoader = DataLoader(prodImgsPathFile,prodLabelsFile,testImgsPathFile,testLabelsFile,doCrop,doFlip,trainMode)

-- Save product image features
local prodImgFeats = torch.zeros(dataLoader.numProdImgs,2048)
local prodImgIdx = 1
for objIdx = 1,dataLoader.numProdClasses do
    for imgIdx = 1,#dataLoader.prodImgPaths[objIdx] do
        prodImgFeats[prodImgIdx] = dataLoader.prodImgFeats[objIdx][imgIdx]
        prodImgIdx = prodImgIdx+1
    end
end

local resultsFile = hdf5.open(paths.concat(snapshotsFolder,'results-'..snapshotName..'.h5'), 'w')
resultsFile:write('prodFeat', prodImgFeats:float())

-- Save training image features and class confidence values
local testImgPaths = getLinesFromFile(testImgsPathFile)
local testImgFeats = torch.ones(#testImgPaths,2048)
local testImgClassConf = torch.zeros(#testImgPaths,dataLoader.numTrainClasses)
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
    testImgFeats[imgIdx] = imgEncoder:forward(testImg:cuda()):float()[1]
end

resultsFile:write('testFeat', testImgFeats:float())
resultsFile:close()

print('Finished.')
