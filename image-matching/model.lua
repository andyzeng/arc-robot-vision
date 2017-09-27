require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'


function getColorClassMultiProdSkipConnModel(numObjClasses,numTypeClasses)

    -- Load 2 ResNet-101 pre-trained on ImageNet as RGB-D tower
    local rgbTrunk = torch.load('resnet-50.t7')
    rgbTrunk:remove(11)
    rgbTrunk:insert(nn.Normalize(2))

    -- Load 3 copies of ResNet-101 in a triplet model:
    -- First trunk: tote RGB-D image of anchor object
    -- Second trunk: product RGB image of anchor object
    -- Third trunk: tote RGB-D image of another object different from anchor object
    local tripletTrunk = nn.ParallelTable()
    local prodImgTower = nn.Sequential():add(nn.Identity)
    local toteImgTower = rgbTrunk:clone()
    tripletTrunk:add(toteImgTower) 
    tripletTrunk:add(prodImgTower)
    tripletTrunk:add(toteImgTower:clone('weight','bias','gradWeight','gradBias')) 

    -- Build classification joint (object class)
    local classifyObjJoint = nn.Sequential():add(nn.SelectTable(1))
    classifyObjJoint:add(nn.Linear(2048,512)):add(nn.BatchNormalization(512)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyObjJoint:add(nn.Linear(512,128)):add(nn.BatchNormalization(128)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyObjJoint:add(nn.Linear(128,numObjClasses))

    -- Build classification joint (object type)
    local classifyTypeJoint = nn.Sequential():add(nn.SelectTable(1))
    classifyTypeJoint:add(nn.Linear(2048,512)):add(nn.BatchNormalization(512)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyTypeJoint:add(nn.Linear(512,128)):add(nn.BatchNormalization(128)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyTypeJoint:add(nn.Linear(128,numTypeClasses))

    -- Build pairwise sample distance model (L2)
    local posDistJoint = nn.Sequential() -- Similar sample distance w.r.t anchor sample
    posDistJoint:add(nn.NarrowTable(1,2)):add(nn.PairwiseDistance(2))
    local negDistJoint = nn.Sequential() -- Different sample distance w.r.t anchor sample
    negDistJoint:add(nn.NarrowTable(2,2)):add(nn.PairwiseDistance(2))
    local distJoint = nn.ConcatTable():add(posDistJoint):add(negDistJoint):add(classifyObjJoint):add(classifyTypeJoint)

    -- Build complete model
    local model = nn.Sequential()
    model:add(tripletTrunk)
    model:add(distJoint)

    -- Defines triplet loss from "Deep Metric Learning using Triplet Network" http://arxiv.org/abs/1412.6622
    local criterionDist = nn.DistanceRatioCriterion(true):cuda()

    local criterionClass = nn.CrossEntropyCriterion():cuda()

    local criterionType = nn.CrossEntropyCriterion():cuda()

    return model,criterionDist,criterionClass,criterionType
end

function getColorClassMultiProdModel(numClasses)

    -- Load 2 ResNet-101 pre-trained on ImageNet as RGB-D tower
    local rgbTrunk = torch.load('resnet-50.t7')
    rgbTrunk:remove(11)
    rgbTrunk:insert(nn.Normalize(2))

    -- Load 3 copies of ResNet-101 in a triplet model:
    -- First trunk: tote RGB-D image of anchor object
    -- Second trunk: product RGB image of anchor object
    -- Third trunk: tote RGB-D image of another object different from anchor object
    local tripletTrunk = nn.ParallelTable()
    local prodImgTower = nn.Sequential():add(nn.Identity)
    local toteImgTower = rgbTrunk:clone()
    tripletTrunk:add(toteImgTower) 
    tripletTrunk:add(prodImgTower)
    tripletTrunk:add(toteImgTower:clone('weight','bias','gradWeight','gradBias')) 

    -- Build classification joint
    local classifyJoint = nn.Sequential():add(nn.SelectTable(1))
    classifyJoint:add(nn.Linear(2048,512)):add(nn.BatchNormalization(512)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyJoint:add(nn.Linear(512,128)):add(nn.BatchNormalization(128)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyJoint:add(nn.Linear(128,numClasses))

    -- Build pairwise sample distance model (L2)
    local posDistJoint = nn.Sequential() -- Similar sample distance w.r.t anchor sample
    posDistJoint:add(nn.NarrowTable(1,2)):add(nn.PairwiseDistance(2))
    local negDistJoint = nn.Sequential() -- Different sample distance w.r.t anchor sample
    negDistJoint:add(nn.NarrowTable(2,2)):add(nn.PairwiseDistance(2))
    local distJoint = nn.ConcatTable():add(posDistJoint):add(negDistJoint):add(classifyJoint)

    -- Build complete model
    local model = nn.Sequential()
    model:add(tripletTrunk)
    model:add(distJoint)

    -- Defines triplet loss from "Deep Metric Learning using Triplet Network" http://arxiv.org/abs/1412.6622
    local criterionDist = nn.DistanceRatioCriterion(true):cuda()

    local criterionClass = nn.CrossEntropyCriterion():cuda()

    return model,criterionDist,criterionClass
end

function getColorNoClassFullProdModel(isSharedWeights)

    -- Load 2 ResNet-101 pre-trained on ImageNet as RGB-D tower
    local rgbTrunk = torch.load('resnet-50.t7')
    rgbTrunk:remove(11)
    rgbTrunk:insert(nn.Normalize(2))

    -- Load 3 copies of ResNet-101 in a triplet model:
    -- First trunk: tote RGB-D image of anchor object
    -- Second trunk: product RGB image of anchor object
    -- Third trunk: tote RGB-D image of another object different from anchor object
    local tripletTrunk = nn.ParallelTable()
    local prodImgTower = rgbTrunk:clone()
    local toteImgTower = rgbTrunk:clone()
    tripletTrunk:add(toteImgTower) 
    -- recursiveModelFreeze(prodImgTower) -- Freeze weights of second tower
    if isSharedWeights then
        tripletTrunk:add(toteImgTower:clone('weight','bias','gradWeight','gradBias')) 
    else
        tripletTrunk:add(prodImgTower)
    end
    tripletTrunk:add(toteImgTower:clone('weight','bias','gradWeight','gradBias')) 

    -- Build pairwise sample distance model (L2)
    local posDistJoint = nn.Sequential() -- Similar sample distance w.r.t anchor sample
    posDistJoint:add(nn.NarrowTable(1,2)):add(nn.PairwiseDistance(2))
    local negDistJoint = nn.Sequential() -- Different sample distance w.r.t anchor sample
    negDistJoint:add(nn.NarrowTable(2,2)):add(nn.PairwiseDistance(2))
    local distJoint = nn.ConcatTable():add(posDistJoint):add(negDistJoint)

    -- Build complete model
    local model = nn.Sequential()
    model:add(tripletTrunk)
    model:add(distJoint)

    -- Defines triplet loss from "Deep Metric Learning using Triplet Network" http://arxiv.org/abs/1412.6622
    local criterionDist = nn.DistanceRatioCriterion(true):cuda()

    return model,criterionDist
end

function getColorNoClassMultiProdModel()

    -- Load 2 ResNet-101 pre-trained on ImageNet as RGB-D tower
    local rgbTrunk = torch.load('resnet-50.t7')
    rgbTrunk:remove(11)
    rgbTrunk:insert(nn.Normalize(2))

    -- Load 3 copies of ResNet-101 in a triplet model:
    -- First trunk: tote RGB-D image of anchor object
    -- Second trunk: product RGB image of anchor object
    -- Third trunk: tote RGB-D image of another object different from anchor object
    local tripletTrunk = nn.ParallelTable()
    local prodImgTower = nn.Sequential():add(nn.Identity)
    local toteImgTower = rgbTrunk:clone()
    tripletTrunk:add(toteImgTower) 
    tripletTrunk:add(prodImgTower)
    tripletTrunk:add(toteImgTower:clone('weight','bias','gradWeight','gradBias')) 

    -- Build pairwise sample distance model (L2)
    local posDistJoint = nn.Sequential() -- Similar sample distance w.r.t anchor sample
    posDistJoint:add(nn.NarrowTable(1,2)):add(nn.PairwiseDistance(2))
    local negDistJoint = nn.Sequential() -- Different sample distance w.r.t anchor sample
    negDistJoint:add(nn.NarrowTable(2,2)):add(nn.PairwiseDistance(2))
    local distJoint = nn.ConcatTable():add(posDistJoint):add(negDistJoint)

    -- Build complete model
    local model = nn.Sequential()
    model:add(tripletTrunk)
    model:add(distJoint)

    -- Defines triplet loss from "Deep Metric Learning using Triplet Network" http://arxiv.org/abs/1412.6622
    local criterionDist = nn.DistanceRatioCriterion(true):cuda()

    return model,criterionDist
end

-- Color model with classification loss
function getColorClassModel()

    -- Load 2 ResNet-101 pre-trained on ImageNet as RGB-D tower
    local rgbTrunk = torch.load('resnet-50.t7')
    rgbTrunk:remove(11)
    rgbTrunk:insert(nn.Normalize(2))

    -- Load 3 copies of ResNet-101 in a triplet model:
    -- First trunk: tote RGB-D image of anchor object
    -- Second trunk: product RGB image of anchor object
    -- Third trunk: tote RGB-D image of another object different from anchor object
    local tripletTrunk = nn.ParallelTable()
    local prodImgTower = rgbTrunk:clone()
    local toteImgTower = rgbTrunk:clone()
    tripletTrunk:add(toteImgTower) 
    recursiveModelFreeze(prodImgTower) -- Freeze weights of second tower
    tripletTrunk:add(prodImgTower)
    tripletTrunk:add(toteImgTower:clone('weight','bias','gradWeight','gradBias')) 

    -- Build classification joint
    local classifyJoint = nn.Sequential():add(nn.SelectTable(1))
    classifyJoint:add(nn.Linear(2048,512)):add(nn.BatchNormalization(512)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyJoint:add(nn.Linear(512,128)):add(nn.BatchNormalization(128)):add(nn.ReLU()):add(nn.Dropout(0.5))
    classifyJoint:add(nn.Linear(128,30))

    -- Build pairwise sample distance model (L2)
    local posDistJoint = nn.Sequential() -- Similar sample distance w.r.t anchor sample
    posDistJoint:add(nn.NarrowTable(1,2)):add(nn.PairwiseDistance(2))
    local negDistJoint = nn.Sequential() -- Different sample distance w.r.t anchor sample
    negDistJoint:add(nn.NarrowTable(2,2)):add(nn.PairwiseDistance(2))
    local distJoint = nn.ConcatTable():add(posDistJoint):add(negDistJoint):add(classifyJoint)

    -- Build complete model
    local model = nn.Sequential()
    model:add(tripletTrunk)
    model:add(distJoint)

    -- Defines triplet loss from "Deep Metric Learning using Triplet Network" http://arxiv.org/abs/1412.6622
    local criterionDist = nn.DistanceRatioCriterion(true):cuda()

    local criterionClass = nn.CrossEntropyCriterion():cuda()

    return model,criterionDist,criterionClass
end

function getRGBDModel()

    -- Load 2 ResNet-101 pre-trained on ImageNet as RGB-D tower
    local rgbTrunk = torch.load('resnet-50.t7')
    rgbTrunk:remove(11)
    rgbTrunk:insert(nn.Normalize(2))
    local dTrunk = rgbTrunk:clone()
    local rgbdParallel = nn.ParallelTable():add(rgbTrunk):add(dTrunk)
    local rgbdTrunk = nn.Sequential():add(rgbdParallel):add(nn.JoinTable(2))
    rgbdTrunk:add(nn.Linear(4096,2048)):add(nn.Normalize(2))

    -- Load 3 copies of ResNet-101 in a triplet model:
    -- First trunk: tote RGB-D image of anchor object
    -- Second trunk: product RGB image of anchor object
    -- Third trunk: tote RGB-D image of another object different from anchor object
    local tripletTrunk = nn.ParallelTable()
    local prodImgTower = rgbTrunk:clone()
    local toteImgTower = rgbdTrunk:clone()
    tripletTrunk:add(toteImgTower) 
    recursiveModelFreeze(prodImgTower) -- Freeze weights of second tower
    tripletTrunk:add(prodImgTower)
    tripletTrunk:add(toteImgTower:clone('weight','bias','gradWeight','gradBias')) 

    -- Build pairwise sample distance model (L2)
    local posDistJoint = nn.Sequential() -- Similar sample distance w.r.t anchor sample
    posDistJoint:add(nn.NarrowTable(1,2)):add(nn.PairwiseDistance(2))
    local negDistJoint = nn.Sequential() -- Different sample distance w.r.t anchor sample
    negDistJoint:add(nn.NarrowTable(2,2)):add(nn.PairwiseDistance(2))
    local distJoint = nn.ConcatTable():add(posDistJoint):add(negDistJoint)

    -- Build complete model
    local model = nn.Sequential()
    model:add(tripletTrunk)
    model:add(distJoint)

    -- Defines triplet loss from "Deep Metric Learning using Triplet Network" http://arxiv.org/abs/1412.6622
    local criterion = nn.DistanceRatioCriterion(true):cuda()

    return model,criterion
end

function getPretrainedRGBDModel()

    -- Load pretrained models
    local rgbModel = torch.load('snapshots-v9-color/snapshot_12000.net')
    local rgbTrunk = rgbModel:get(1):get(1):clone()
    local dModel = torch.load('snapshots-v9-depth/snapshot_24000.net')
    local dTrunk = dModel:get(1):get(1):clone()

    -- Build RGB-D joint
    local rgbdParallel = nn.ParallelTable():add(rgbTrunk):add(dTrunk)
    local rgbdTrunk = nn.Sequential():add(rgbdParallel):add(nn.JoinTable(2))
    rgbdTrunk:add(nn.Linear(4096,2048)):add(nn.Normalize(2))

    -- Load 3 copies of ResNet-101 in a triplet model:
    -- First trunk: tote RGB-D image of anchor object
    -- Second trunk: product RGB image of anchor object
    -- Third trunk: tote RGB-D image of another object different from anchor object
    local tripletTrunk = nn.ParallelTable()
    local prodImgTower = rgbTrunk:clone()
    local toteImgTower = rgbdTrunk:clone()
    recursiveModelFreeze(toteImgTower)
    tripletTrunk:add(toteImgTower) 
    recursiveModelFreeze(prodImgTower) -- Freeze weights of second tower
    tripletTrunk:add(prodImgTower)
    local toteImgTowerClone = toteImgTower:clone('weight','bias','gradWeight','gradBias')
    recursiveModelFreeze(toteImgTowerClone)
    tripletTrunk:add(toteImgTowerClone) 

    -- Build pairwise sample distance model (L2)
    local posDistJoint = nn.Sequential() -- Similar sample distance w.r.t anchor sample
    posDistJoint:add(nn.NarrowTable(1,2)):add(nn.PairwiseDistance(2))
    local negDistJoint = nn.Sequential() -- Different sample distance w.r.t anchor sample
    negDistJoint:add(nn.NarrowTable(2,2)):add(nn.PairwiseDistance(2))
    local distJoint = nn.ConcatTable():add(posDistJoint):add(negDistJoint)

    -- Build complete model
    local model = nn.Sequential()
    model:add(tripletTrunk)
    model:add(distJoint)

    -- Defines triplet loss from "Deep Metric Learning using Triplet Network" http://arxiv.org/abs/1412.6622
    local criterion = nn.DistanceRatioCriterion(true):cuda()

    return model,criterion
end