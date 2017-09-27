% Testing script for running a baseline algorithm to detect anti-podal
% parallel-jaw grasps over the testing split of the grasping dataset.
% Grasps are visualized by lines where the end-points are finger locations.

% User options (change me)
dataPath = '../data';  % Path to parallel-jaw grasping dataset

% Parse test split from dataset
testSplit = textread(fullfile(dataPath,'test-split.txt'),'%s','delimiter','\n');

% Loop through all test samples and run baseline algorithm
results = cell(length(testSplit),1);
for sampleIdx = 1:length(testSplit)
    fprintf('Testing: %d/%d\n',sampleIdx,length(testSplit));
    sampleName = testSplit{sampleIdx};
    
    % Load sample heightmap from dataset
    heightmap = imread(fullfile(dataPath,'heightmap-depth',sprintf('%s.png',sampleName)));
    heightmap = double(heightmap)./10000; % Scale to meters
    heightmap = heightmap(13:212,11:310); % Remove extra padding
    imshow(heightmap);
    
    % Run baseline grasp detection on test sample heightmaps
    [graspPredictions,flushGraspPredictions] = predict(heightmap);
    results{sampleIdx} = [graspPredictions;flushGraspPredictions];
    pause(0.1);
end

% Save all output grasp detections to results
save('results.mat','results');













