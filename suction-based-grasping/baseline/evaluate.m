% Script for evaluating suction-based grasping affordance predictions with
% baseline algorithm against manually annotated dataset (% precision)

% User options (change me)
dataPath = '../data'; % Path to suction-based grasping dataset

% Parse test split from dataset
testSplit = textread(fullfile(dataPath,'test-split.txt'),'%s','delimiter','\n');

% Load baseline suction prediction results
load('results.mat');

% Loop through all test samples and evaluate baseline suction prediction
% results against ground truth manual annotations
sumTP = 0; sumFP = 0; sumTN = 0; sumFN = 0;
for sampleIdx = 1:length(testSplit)
    sampleName = testSplit{sampleIdx};
    sampleResult = results{sampleIdx};

    % Load ground truth manual annotations for suction affordances
    % 0 - negative, 128 - positive, 255 - neutral (no loss)
    sampleLabel = imread(fullfile(dataPath,'label',sprintf('%s.png',sampleName)));
    
    % Suction affordance threshold
    % threshold = 0.5; % Confidence threshold based
    % threshold = prctile(sampleResult(:),99); % Top 1%
    threshold = max(sampleResult(:)) - 0.0001; % Top 1 prediction

    % Compute errors
    sampleTP = (sampleResult > threshold) & (sampleLabel == 128);
    sampleFP = (sampleResult > threshold) & (sampleLabel == 0);
    sampleTN = (sampleResult <= threshold) & (sampleLabel == 0);
    sampleFN = (sampleResult <= threshold) & (sampleLabel == 128);
    sumTP = sumTP + sum(sampleTP(:));
    sumFP = sumFP + sum(sampleFP(:));
    sumTN = sumTN + sum(sampleTN(:));
    sumFN = sumFN + sum(sampleFN(:));
end

% Compute total pixel-wise precision/recall over all test samples
precision = sumTP/(sumTP + sumFP);
recall = sumTP/(sumTP + sumFN);
fprintf('Precision: %f\nRecall: %f\n',precision,recall);








