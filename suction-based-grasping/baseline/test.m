% Testing script for running a baseline algorithm to predict
% suction-based affordances over the testing split of the grasping dataset

% User options (change me)
dataPath = '../data'; % Path to suction-based grasping dataset

% Parse test split from dataset
testSplit = textread(fullfile(dataPath,'test-split.txt'),'%s','delimiter','\n');

% Loop through all test samples and run baseline algorithm
results = cell(length(testSplit),1);
for sampleIdx = 1:length(testSplit)
    fprintf('Testing: %d/%d\n',sampleIdx,length(testSplit));
    sampleName = testSplit{sampleIdx};
    
    % Load RGB-D image data and camera intrinsics for input and background
    inputColor = imread(fullfile(dataPath,'color-input',sprintf('%s.png',sampleName)));
    inputDepth = double(imread(fullfile(dataPath,'depth-input',sprintf('%s.png',sampleName))))./10000;
    backgroundColor = imread(fullfile(dataPath,'color-background',sprintf('%s.png',sampleName)));
    backgroundDepth = double(imread(fullfile(dataPath,'depth-background',sprintf('%s.png',sampleName))))./10000;
    cameraIntrinsics = dlmread(fullfile(dataPath,'camera-intrinsics',sprintf('%s.txt',sampleName)));
    
    % % Fill in depth holes
    % addpath(genpath('../external/bxf'));
    % filledDepth = fill_depth_cross_bf(imresize(inputColor,0.25), imresize(inputDepth,0.25,'nearest')); 
    % filledDepth = imresize(filledDepth,4);
    % inputDepth(inputDepth==0) = filledDepth(inputDepth==0);

    % Run baseline suction prediction to get dense suction-based grasping affordance scores
    suctionScores = predict(inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics);
    results{sampleIdx} = suctionScores;
    
    % Display input images and output suction-based grasping affordance scores
    subplot(1,3,1); imshow(inputColor); axis equal; title('Color'); 
    subplot(1,3,2); imagesc(inputDepth); axis equal; title('Depth');
    subplot(1,3,3); imagesc(suctionScores); axis equal; title('Prediction');
    pause(0.1);
end

% Save all output suction-based grasping affordance scores to results
save('results.mat','results');
