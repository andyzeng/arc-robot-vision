clear all; close all;

% Model to use for classifying known vs novel and for recognizing known objects
kNetPath = fullfile('snapshots-with-class','results-snapshot-170000.h5');

% Model to use for recognizing novel objects
nNetPath = fullfile('snapshots-no-class','results-snapshot-8000.h5');

% Get ground truth labels and list of objects in bin
testImgsLabels = dlmread('data/test-labels.txt');
prodImgLabel = sort(dlmread('data/test-product-labels.txt'));
testOtherObjList = dlmread('data/test-other-objects-list.txt');

% Get training and testing class names
trainImgPath = 'data/train-imgs';
trainImgList = dir(trainImgPath);
trainImgList = trainImgList(3:end);
trainClasses = {};
for objIdx = 1:length(trainImgList)
    trainClasses{length(trainClasses)+1} = lower(trainImgList(objIdx).name);
end
testImgPath = 'data/test-item-data';
testImgList = dir(testImgPath);
testImgList = testImgList(3:end);
testClasses = {};
for objIdx = 1:length(testImgList)
    testClasses{length(testClasses)+1} = lower(testImgList(objIdx).name);
end

% Get predictions for classifying known vs novel with K-net
prodImgsFeats = hdf5read(kNetPath,'prodFeat');
prodImgsFeats = prodImgsFeats';
testImgsFeats = hdf5read(kNetPath,'testFeat');
testImgsFeats = testImgsFeats';

% Check which objects are known vs novel
prodIsKnownObj = zeros(size(prodImgsFeats,1),1);
testIsKnownObj = zeros(size(testImgsFeats,1),1);
for objIdx = 1:length(testClasses)
     trainClassIdx = cellfun(@(x) strcmp(testClasses{objIdx},x),trainClasses);
     if any(trainClassIdx)
         prodIsKnownObj(find(prodImgLabel == objIdx)) = 1;
         testIsKnownObj(find(testImgsLabels == objIdx)) = 1;
     end
end

predNnDist = zeros(size(testImgsLabels));
for testImgIdx = 1:size(testImgsFeats,1)
    testImgFeat = testImgsFeats(testImgIdx,:);
    featDists = sqrt(sum((repmat(testImgFeat,size(prodImgsFeats,1),1)-prodImgsFeats).^2,2));
    featDists = [prodImgLabel,featDists];
    validProdImgInd = arrayfun(@(x) any(testOtherObjList(testImgIdx,:) == x),prodImgLabel);
    sortedFeatDists = sortrows(featDists(find(validProdImgInd),:),2);
    predNnDist(testImgIdx) = min(sortedFeatDists(:,2));
end
bestKnownNovelAcc = 0;
bestKnownNovelThreshold = 0;
for threshold = 0:0.01:1.2
    knownNovelAcc = sum((predNnDist > threshold) == ~testIsKnownObj)/length(testIsKnownObj);
    if bestKnownNovelAcc < knownNovelAcc
        bestKnownNovelAcc = knownNovelAcc;
        bestKnownNovelThreshold = threshold;
    end
end
% knetPredIsKnownObj = predNnDist < bestKnownNovelThreshold;

% Load model for guessing novel objects (N-net)
nNetImgsFeats = hdf5read(nNetPath,'testFeat');
nNetImgsFeats = nNetImgsFeats';

% Get final predictions
predLabels = zeros(size(testImgsFeats,1));
for testImgIdx = 1:size(testImgsFeats,1)
    testImgFeat = testImgsFeats(testImgIdx,:);
    featDists = sqrt(sum((repmat(testImgFeat,size(prodImgsFeats,1),1)-prodImgsFeats).^2,2));
    featDists = [prodImgLabel,featDists];
    validProdImgInd = arrayfun(@(x) any(testOtherObjList(testImgIdx,:) == x),prodImgLabel);
    sortedFeatDists = sortrows(featDists(find(validProdImgInd),:),2);
    nnDist = min(sortedFeatDists(:,2));
    
    % If nearest neighbor distance is small enough, assume object is known and use K-net
    if nnDist < bestKnownNovelThreshold
        sortedFeatDists = sortrows(featDists(find(validProdImgInd & prodIsKnownObj),:),2);
        predLabels(testImgIdx) = sortedFeatDists(1,1);
        
    % If nearest neighbor distance is far, assume object is unknown and use N-net
    else
        testImgFeat = nNetImgsFeats(testImgIdx,:);
        featDists = sqrt(sum((repmat(testImgFeat,size(prodImgsFeats,1),1)-prodImgsFeats).^2,2));
        featDists = [prodImgLabel,featDists];
        validProdImgInd = arrayfun(@(x) any(testOtherObjList(testImgIdx,:) == x),prodImgLabel);
        sortedFeatDists = sortrows(featDists(find(validProdImgInd & ~prodIsKnownObj),:),2);
        predLabels(testImgIdx) = sortedFeatDists(1,1);
    end
end

% Compute and report all accuracies
oldObjImgIdx = find(testIsKnownObj);
newObjImgIdx = find(~testIsKnownObj);
oldObjAcc = sum(testImgsLabels(oldObjImgIdx) == predLabels(oldObjImgIdx,1))/length(oldObjImgIdx);
newObjAcc = sum(testImgsLabels(newObjImgIdx) == predLabels(newObjImgIdx,1))/length(newObjImgIdx);
avgObjAcc = sum(testImgsLabels == predLabels(:,1),1)/length(testImgsLabels);
fprintf('Old object accuracy: %f\nNovel object accuracy: %f\nMixed object accuracy: %f\nKnown vs novel classification accuracy: %f\n',oldObjAcc,newObjAcc,avgObjAcc,bestKnownNovelAcc);





