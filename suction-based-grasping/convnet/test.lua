require 'cutorch';
require 'cunn';
require 'cudnn';
require 'image';
require 'nn';
require 'nnx';
require 'optim';
require 'hdf5';
require 'util.lua'
require 'model.lua'

-- Default user options
options = {
  dataPath = '../data',
  testImgList = '../data/test-split.txt',
  modelPath = 'suction-based-grasping-snapshot-10001.t7',
  resultsPath = 'evaluation-results.h5',
  outputScale = 1/8,
  imgHeight =  480,
  imgWidth = 640
}

-- Parse user options from command line (i.e. modelPath=<model.t7> th test.lua)
for k,v in pairs(options) do options[k] = tonumber(os.getenv(k)) or os.getenv(k) or options[k] end

-- Read paths for data samples
local testImgPaths = getLinesFromFile(options.testImgList)

-- Set RNG seed
math.randomseed(os.time())

-- Load trained model and set to testing (evaluation) mode
print('Loading model: '..options.modelPath)
local model = torch.load(options.modelPath)
model:add(cudnn.SpatialSoftMax())
model = model:cuda()
model:evaluate()

-- Define dataset average constants (r,g,b)
local mean = {0.485,0.456,0.406}
local std = {0.229,0.224,0.225}

-- Initialize empty tensors for input and output
local input  = {torch.Tensor(1, 3, options.imgHeight, options.imgWidth),torch.Tensor(1, 3, options.imgHeight, options.imgWidth)}
local results = torch.Tensor(1,3,options.imgHeight*options.outputScale,options.imgWidth*options.outputScale):float()

-- Loop through all test images and compute forward passes for each image
for sampleIdx = 1,#testImgPaths do
    print('Testing: '..sampleIdx..'/'..#testImgPaths);

    -- Load and pre-process color image (24-bit RGB PNG)
    local colorImg = image.load(paths.concat(options.dataPath,'color-input',testImgPaths[sampleIdx]..'.png'))
    for c=1,3 do
        colorImg[c]:add(-mean[c])
        colorImg[c]:div(std[c])
    end

    -- Load and pre-process depth image (16-bit PNG depth in deci-millimeters)
    local depth = image.load(paths.concat(options.dataPath,'depth-input',testImgPaths[sampleIdx]..'.png'))
    depth = depth*65536/10000
    depth = depth:clamp(0.0,1.2) -- Depth range of Intel RealSense SR300
    local depthImg = depth:cat(depth,1):cat(depth,1)
    for c=1,3 do
        depthImg[c]:add(-mean[c])
        depthImg[c]:div(std[c])
    end

    -- Copy images into input tensors and convert them to CUDA
    input[1]:copy(colorImg:reshape(1, 3, options.imgHeight, options.imgWidth))
    input[2]:copy(depthImg:reshape(1, 3, options.imgHeight, options.imgWidth))
    input[1] = input[1]:cuda()
    input[2] = input[2]:cuda()

    -- Compute forward pass
    local output = model:forward(input)
    if sampleIdx == 1 then
        results = output:float()
    else
        results = results:cat(output:float(),1)
    end
end

-- Save output test results
print('Saving results to: '..options.resultsPath)
local testResultsFile = hdf5.open(options.resultsPath, 'w')
testResultsFile:write('results', results:float())
testResultsFile:close()
