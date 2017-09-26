require 'cutorch'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'optim'

-- Load some useful functions from SharpMask/DeepMask
paths.dofile('SpatialSymmetricPadding.lua')
local utils = paths.dofile('modelUtils.lua')

-- Build fully convolutional RGB-D ResNet-101 
function getModel(options)

    -- Load ResNet-101 pre-trained on ImageNet
    local rgbTrunk = torch.load('resnet-101.t7')

    -- Remove BN
    utils.BNtoFixed(rgbTrunk, true)

    -- Remove FC layers
    rgbTrunk:remove()
    rgbTrunk:remove()
    rgbTrunk:remove()
    rgbTrunk:remove()

    -- Add depth network
    local dTrunk = rgbTrunk:clone()

    -- Build RGB-D joint
    local rgbdParallel = nn.ParallelTable():add(rgbTrunk):add(dTrunk)
    local model = nn.Sequential():add(rgbdParallel):add(nn.JoinTable(2))

    -- Add standard convolution layers
    model:add(cudnn.SpatialConvolution(2048,512,1,1,1,1))
    model:add(cudnn.SpatialConvolution(512,128,1,1,1,1))
    model:add(cudnn.SpatialConvolution(128,options.nClass,1,1,1,1))
    model:add(nn.SpatialUpSamplingBilinear(2))

    -- Use SymmetricPadding
    utils.updatePadding(model, nn.SpatialSymmetricPadding)

    -- Set class weights (unlabeled regions = 0)
    local classWeights = torch.ones(3)
    classWeights[3] = 0

    local criterion = cudnn.SpatialCrossEntropyCriterion(classWeights)

    return model,criterion
end
