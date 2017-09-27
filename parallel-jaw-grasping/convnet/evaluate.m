% Script for evaluating parallel-jaw grasping affordance predictions
% against manually annotated dataset (% precision)

% User options (change me)
dataPath = '../data';                   % Path to unprocessed parallel-jaw grasping dataset
resultsFile = 'evaluation-results.h5';  % HDF5 ConvNet output file from running test.lua

% Path to processed grasp labels dataset
labelPath = fullfile('training','label');

% Parse test split from dataset
testSplit = textread(fullfile(dataPath,'test-split.txt'),'%s','delimiter','\n');

% Load ConvNet affordance prediction results
results = hdf5read(resultsFile,'results');
results = permute(results,[2,1,3,4]);

% Loop through all test samples and evaluate ConvNet paralle-jaw grasping
% affordance predictions against manual annotations
sumTP = 0; sumFP = 0; sumTN = 0; sumFN = 0;
for sampleIdx = 1:length(testSplit)
    fprintf('Testing: %d/%d\n',sampleIdx,length(testSplit));
    sampleName = testSplit{sampleIdx};
    
    % Load convnet predictions
    sampleResult = results(:,:,2,((sampleIdx-1)*16+1):(sampleIdx*16));
    
    % Evaluate for each grasp angle (16 rotations)
    for rotIdx = 1:16
        samplePrediction = sampleResult(:,:,:,rotIdx);
        
        % Load ground truth manual annotations for affordances
        % 0 - negative, 128 - positive, 255 - neutral (no loss)
        sampleLabel = imread(fullfile(labelPath,sprintf('%s-%02d.png',sampleName,rotIdx-1)));
        if sum(sampleLabel(:) < 255) == 0
            continue;
        end
        
        % Affordance threshold
	    % threshold = 0.5; % Confidence threshold based
        % threshold = prctile(samplePrediction(:),90); % Top 1%
        threshold = max(samplePrediction(:)) - 0.0001; % Top 1 prediction

        % Compute errors
        sampleTP = (samplePrediction > threshold) & (sampleLabel == 128);
        sampleFP = (samplePrediction > threshold) & (sampleLabel == 0);
        sampleTN = (samplePrediction <= threshold) & (sampleLabel == 0);
        sampleFN = (samplePrediction <= threshold) & (sampleLabel == 128);
        sumTP = sumTP + sum(sampleTP(:));
        sumFP = sumFP + sum(sampleFP(:));
        sumTN = sumTN + sum(sampleTN(:));
        sumFN = sumFN + sum(sampleFN(:));
    end
end

% Compute total pixel-wise precision over all test samples
precision = sumTP/(sumTP + sumFP);
fprintf('Average precision: %f\n',precision);








