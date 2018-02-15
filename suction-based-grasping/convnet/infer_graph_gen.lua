require 'cutorch';
require 'cunn';
require 'cudnn'
require 'image'
require 'nn'
require 'nnx'
require 'optim'
require 'hdf5'
require 'util'
require 'DataLoader'
require 'model'

-- Default user options
options = {
  imgColorPath = 'demo/test-image.color.png',
  imgDepthPath = 'demo/test-image.depth.png',
  modelPath = 'suction-based-grasping-snapshot-10001.t7',
  resultsPath = 'demo/results.h5',
  outputScale = 1/8,
  imgHeight =  480,
  imgWidth = 640
}

-- Parse user options from command line (i.e. imgColorPath=<image.png> imgDepthPath=<image.png> modelPath=<model.t7> th infer.lua)
for k,v in pairs(options) do options[k] = tonumber(os.getenv(k)) or os.getenv(k) or options[k] end

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

-- Load and pre-process color image (24-bit RGB PNG)
print('Pre-processing color image: '..options.imgColorPath)
local colorImg = image.load(options.imgColorPath)
for c=1,3 do
    colorImg[c]:add(-mean[c])
    colorImg[c]:div(std[c])
end

-- Load and pre-process depth image (16-bit PNG depth in deci-millimeters)
print('Pre-processing depth image: '..options.imgDepthPath)
local depth = image.load(options.imgDepthPath)
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

-- Plot

--require 'cunn'
--require 'nn'
--require 'cudnn'
--require 'inn'
--models = require 'optnet.models'
--modelname = 'googlenet'
--net, input = models[modelname]()

--net = torch.load('suction-based-grasping-snapshot-10001.t7')
generateGraph = require 'optnet.graphgen'

-- visual properties of the generated graph
-- follows graphviz attributes
graphOpts = {
displayProps =  {shape='ellipse',fontsize=14, style='solid'},
nodeData = function(oldData, tensor)
  --return oldData .. '\n' .. 'Size: '.. tensor:size()
  return tensor:size()
end
}

g = generateGraph(model, input, graphOpts)

graph.dot(g,'model_arch','model_arch')

-- Compute forward pass
print('Computing forward pass...')
local output = model:forward(input)

-- Save output test results
print('Saving results to: '..options.resultsPath)
results = output:float()
local resultsFile = hdf5.open(options.resultsPath, 'w')
resultsFile:write('results', results:float())
resultsFile:close()
print('Done.')