require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
-- require 'DistanceRatioCriterion' (uncomment if DistanceRatioCriterion is not found)
require 'utils.lua'
require 'model.lua'
require 'DataLoader.lua'

-- Set RNG seed
math.randomseed(os.time())

-- Set training mode to change deep model (see our paper:
-- "Robotic Pick-and-Place of Novel Objects in Clutter
-- with Multi-Affordance Grasping and Cross-Domain Image Matching")
local trainMode = 1
local snapshotsFolder

-- Two-stream + guided-embedding + multi-product-images + auxiliary classification (K-net)
-- This is the best performing model for known objects
if trainMode == 1 then
	cutorch.setDevice(1)
	snapshotsFolder = 'snapshots-with-class'
end

-- Two-stream + guided-embedding + multi-product-images (N-net)
-- This is the best performing model for novel objects
if trainMode == 2 then
	cutorch.setDevice(1)
	snapshotsFolder = 'snapshots-no-class'
end

-- Two-stream + guided-embedding
if trainMode == 3 then
	cutorch.setDevice(1)
	snapshotsFolder = 'snapshots-no-switch'
end

-- Two-stream network without weight sharing
if trainMode == 4 then
	cutorch.setDevice(1)
	snapshotsFolder = 'snapshots-both-train'
end

-- Siamese network with weight sharing
if trainMode == 5 then
	cutorch.setDevice(1)
	snapshotsFolder = 'snapshots-both-train-shared'
end

local doCrop = true; -- crop observed images?
local doFlip = true; -- augment training data with flipping?

-- Create data sampler
print('Loading dataset...')
local prodImgsPathFile = 'data/train-product-imgs.txt'
local prodLabelsFile = 'data/train-product-labels.txt'
local trainImgsPathFile = 'data/train-imgs.txt'
local trainLabelsFile = 'data/train-labels.txt'
local dataLoader = DataLoader(prodImgsPathFile,prodLabelsFile,trainImgsPathFile,trainLabelsFile,doCrop,doFlip,trainMode)

-- Build deep model and set it to training mode
print('Building model...')
print('Number of training classes: '..dataLoader.numTrainClasses)
local model,criterionDist,criterionClass
if trainMode == 1 then
	model,criterionDist,criterionClass = getColorClassMultiProdModel(dataLoader.numTrainClasses)
end
if trainMode == 2 or trainMode == 3 then
	model,criterionDist = getColorNoClassMultiProdModel()
end
if trainMode == 4 then
	model,criterionDist = getColorNoClassFullProdModel(false)
end
if trainMode == 5 then
	model,criterionDist = getColorNoClassFullProdModel(true)
end
model = model:cuda()
model:training()

-- Get model parameters
local params,gradParams = model:getParameters()

-- Start training
for trainIter = 1,10000000 do

	-- Create a mini-batch of training examples
	local input,classLabel = dataLoader:getTrainingBatch()

	for i = 1,dataLoader.batchSize do
		if i == 1 then
			mosaic = input[1][i]:reshape(3,224,224):cat(input[3][i]:reshape(3,224,224),2)
		else
			mosaic = mosaic:cat(input[1][i]:reshape(3,224,224):cat(input[3][i]:reshape(3,224,224),2),3)
		end
	end

	-- Convert input to GPU memory
	input[1] = input[1]:cuda()
	input[2] = input[2]:cuda()
	input[3] = input[3]:cuda()

	local feval = function(x)

	    -- Update model parameters
	    if x ~= params then
	        params:copy(x)
	    end

	    -- Reset gradients
	    gradParams:zero()

		local output
	    if trainMode == 1 or trainMode == 2 or trainMode == 3 then

		    -- Switch anchor product image based on nearest neighbor features
		    local outputFeat = model:get(1):forward(input)

		    -- local typeLabel = classLabel:clone()
		    for sampleIdx = 1,dataLoader.batchSize do
		    	local trainImgFeat = outputFeat[1][sampleIdx]
		    	local minDist = math.huge
		    	local closestProdFeatIdx = 0
		    	local distJoint = nn.Sequential():add(nn.PairwiseDistance(2))
		    	local anchorLabel = classLabel[sampleIdx][1]

		    	if trainMode == 1 or trainMode == 2 then
			    	-- Sample based on closest distance
			    	for prodImgIdx = 1,#dataLoader.prodImgPaths[anchorLabel] do
			    		local prodImgFeat = dataLoader.prodImgFeats[anchorLabel][prodImgIdx]
			    		local featDist = distJoint:forward({trainImgFeat:float(),prodImgFeat})[1]
			    		if featDist < minDist then
			    			minDist = featDist
			    			outputFeat[2][sampleIdx] = prodImgFeat
			    		end
			    	end
			    end

			    if trainMode == 3 then
			    	-- Randomly sample
			    	randProdImgIdx = math.floor(math.random()*(#dataLoader.prodImgPaths[anchorLabel]))+1
			    	outputFeat[2][sampleIdx] = dataLoader.prodImgFeats[anchorLabel][randProdImgIdx]:clone()
			    end
		    end
		    input[2] = outputFeat[2]:clone()

		    -- Finish forward pass to compute distances
		    output = model:get(2):forward(outputFeat)
		end

		if trainMode == 4 or trainMode == 5 then
			output = model:forward(input)
		end

	    -- Compute distance ratio loss
	    local lossDist = criterionDist:forward({output[1],output[2]})
	    local dlossDist = criterionDist:backward({output[1],output[2]})

	    local lossClass,dlossClass
		if trainMode == 1 then
		    -- Compute classification loss (object class)
		    lossClass = criterionClass:forward(output[3],classLabel:cuda())
		    dlossClass = criterionClass:backward(output[3],classLabel:cuda())

		    -- Backward pass
	        model:backward(input,{dlossDist[1],dlossDist[2],dlossClass})
	    end

	    if trainMode == 2 or trainMode == 3 or trainMode == 4 or trainMode == 5 then
	    	model:backward(input,{dlossDist[1],dlossDist[2]})
		end

	    if trainIter%10 == 0 then
			if trainMode == 1 then
		    	print('Training iteration '..trainIter..': '..lossDist..' '..lossClass)
		    end
			if trainMode == 2 or trainMode == 3 or trainMode == 4 or trainMode == 5 then
	    		print('Training iteration '..trainIter..': '..lossDist)
		    end
	    end

	    local loss
		if trainMode == 1 then
	    	loss = lossDist+lossClass
	    end
		if trainMode == 2 or trainMode == 3 or trainMode == 4 or trainMode == 5 then
	    	loss = lossDist
	    end

	    return loss,gradParams
	end

	-- Update model parameters (SGD)
	local config = {learningRate = 0.001,momentum = 0.99}
	optim.sgd(feval,params,config)

	-- Save training snapshot of model
	if trainIter%1000 == 0 then
		local filename = paths.concat(snapshotsFolder,'snapshot-'..trainIter..'.t7')
		os.execute('mkdir -p '..sys.dirname(filename))
		torch.save(filename, model:clearState())
	end
end
