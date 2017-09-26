% Script for evaluating suction-based grasping affordance predictions
% against manually annotated dataset (% precision)

% User options (change me)
dataPath = '../data';                         % Path to dataset
resultsFile = 'evaluation-results.h5';        % HDF5 ConvNet output file from running test.lua

% Parse test split from dataset
testSplit = textread(fullfile(dataPath,'test-split.txt'),'%s','delimiter','\n');

% Load ConvNet suction prediction results
results = hdf5read(resultsFile,'results');
results = permute(results,[2,1,3,4]);

% Loop through all test samples and evaluate ConvNet affordance prediction
% results against manual annotations
sumTP = 0; sumFP = 0; sumTN = 0; sumFN = 0;
for sampleIdx = 1:length(testSplit)
    fprintf('Testing: %d/%d\n',sampleIdx,length(testSplit));
    sampleName = testSplit{sampleIdx};
    
    % Post-process convnet prediction values
    sampleAffordances = imresize(results(:,:,2,sampleIdx),8);
    sampleAffordances(sampleAffordances >= 1) = 0.9999;
    sampleAffordances(sampleAffordances < 0) = 0;
    
    % if 0 % Code using post-processing with depth hole filling
    %     
    %     % Load RGB-D image data and camera intrinsics for input and background
    %     inputColor = imread(fullfile(dataPath,'color-input',sprintf('%s.png',sampleName)));
    %     inputDepth = double(imread(fullfile(dataPath,'depth-input',sprintf('%s.png',sampleName))))./10000;
    %     backgroundColor = imread(fullfile(dataPath,'color-background',sprintf('%s.png',sampleName)));
    %     backgroundDepth = double(imread(fullfile(dataPath,'depth-background',sprintf('%s.png',sampleName))))./10000;
    %     cameraIntrinsics = dlmread(fullfile(dataPath,'camera-intrinsics',sprintf('%s.txt',sampleName)));
    % 
    %     % Fill in depth holes
    %     addpath(genpath('../external/bxf'));
    %     inputDepthFilled = fill_depth_cross_bf(imresize(inputColor,0.25), imresize(inputDepth,0.25,'nearest')); 
    %     inputDepthFilled = imresize(inputDepthFilled,4);
    %     inputDepth(inputDepth==0) = inputDepthFilled(inputDepth==0);
    % 
    %     % Post-process results
    %     sampleAffordances = postprocess(sampleAffordances,inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics);
    % end
    
    % Gaussian smooth affordances
    sampleAffordances = imgaussfilt(sampleAffordances, 7);
        
    % Load ground truth manual annotations for suction affordances
    % 0 - negative, 128 - positive, 255 - neutral (no loss)
    sampleLabel = imread(fullfile(dataPath,'label',sprintf('%s.png',sampleName)));
    
    % Suction affordance threshold
    % threshold = 0.5; % Confidence threshold based
    % threshold = prctile(sampleResult(:),99); % Top 1%
    threshold = max(sampleAffordances(:)) - 0.0001; % Top 1 prediction

    % Compute errors
    sampleTP = (sampleAffordances > threshold) & (sampleLabel == 128);
    sampleFP = (sampleAffordances > threshold) & (sampleLabel == 0);
    sampleTN = (sampleAffordances <= threshold) & (sampleLabel == 0);
    sampleFN = (sampleAffordances <= threshold) & (sampleLabel == 128);
    sumTP = sumTP + sum(sampleTP(:));
    sumFP = sumFP + sum(sampleFP(:));
    sumTN = sumTN + sum(sampleTN(:));
    sumFN = sumFN + sum(sampleFN(:));
end

% Compute total pixel-wise precision over all test samples
precision = sumTP/(sumTP + sumFP);
fprintf('Average precision: %f\n',precision);








