require 'image'
require 'cutorch'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'optim'
require 'model.lua'
require 'util.lua'
require 'DataLoader.lua'

-- Default user options
options = {
  batchSize = 2,
  nClass = 3,
  shuffleData = true,
  dataPath = 'training',
  sampleList = 'training/train-processed-split.txt',
  outputScale = 8,
  snapshotsFolder = 'snapshots',
  gpu = 1,
  imgHeight =  320,
  imgWidth = 320,
  learningRate = 0.001
}

-- Parse user options from command line (i.e. sampleList=<list.txt> th train.lua)
for k,v in pairs(options) do options[k] = tonumber(os.getenv(k)) or os.getenv(k) or options[k] end

-- Set active GPU
cutorch.setDevice(options.gpu)

-- Set RNG seed
math.randomseed(os.time())

-- Initialize data loader
local dataLoader = DataLoader(options)

-- Build model and loss criterion
model,criterion = getModel(options)

-- Initialize empty tensors for input and labels
local input  = {torch.Tensor(options.batchSize, 3, options.imgHeight, options.imgWidth),torch.Tensor(options.batchSize, 3, options.imgHeight, options.imgWidth)}
local label  = torch.Tensor(options.batchSize, options.imgHeight/options.outputScale,options.imgWidth/options.outputScale)

-- Load model to GPU
model = model:cuda()
input[1] = input[1]:cuda()
input[2] = input[2]:cuda()
label = label:cuda()
criterion = criterion:cuda()

-- Set model to training mode
model:training()

-- Start training
local params,gradParams = model:getParameters()
for trainIter = 1,1000000 do -- set maximum number of iterations to an arbitrarily large number

  -- Create a mini-batch of training examples
  local rgbdImgs,labelImgs = dataLoader:getMiniBatch()
  input[1]:copy(rgbdImgs[1])
  input[2]:copy(rgbdImgs[2])
  label:copy(labelImgs)

  local feval = function(x)

    -- Update model parameters
    if x ~= params then
      params:copy(x)
    end

    -- Reset gradients
    gradParams:zero()

    -- Compute forward pass
    local output = model:forward(input)

    -- Compute loss
    local loss = criterion:forward(output,label)

    -- Compute backpropagation
    local dloss_dout = criterion:backward(output,label)
    model:backward(input,dloss_dout)

    print('Training iteration '..trainIter..': '..loss)

    return loss,gradParams
  end

  -- Update model parameters with stochastic gradient descent (SGD)
  config = {learningRate = options.learningRate,momentum = 0.99}
  optim.sgd(feval,params,config)

  -- Save training snapshot of model
  if trainIter%1000 == 1 then
    local filename = paths.concat(options.snapshotsFolder, "snapshot-"..trainIter..".t7")
    os.execute('mkdir -p '..sys.dirname(filename))
    torch.save(filename, model:clearState())
  end
end
