% Script for evaluating anti-podal parallel-jaw grasp detection with
% baseline algorithm against manually annotated dataset (% precision)

% User options (change me)
dataPath = '../data';  % Path to unprocessed grasp prediction dataset

% Path to processed grasp labels dataset
labelDir = fullfile(dataPath,'label');

% Parse test split from dataset
testSplit = textread(fullfile(dataPath,'test-split.txt'),'%s','delimiter','\n');

% Load baseline grasping prediction results
load('results.mat');

% Loop through all test samples and evaluate convnet suction prediction
% results against ground truth manual annotations
sumTP = 0; sumFP = 0; sumTN = 0; sumFN = 0;
for sampleIdx = 1:length(testSplit)
    fprintf('Testing: %d/%d\n',sampleIdx,length(testSplit));
    sampleName = testSplit{sampleIdx};
    sampleResult = results{sampleIdx};
    if ~isempty(sampleResult)
        % sampleResult = sampleResult(sampleResult(:,4)>0.5,:); % Confidence threshold based
        % threshold = prctile(sampleResult(:,4),99) - 0.0001; % Top 1%
        % sampleResult = sampleResult(sampleResult(:,4)>threshold,:); % Top 1%
        [~,maxi] = max(sampleResult(:,4)); % Top 1 prediction
        sampleResult = sampleResult(maxi,:); % Top 1 prediction
        sampleResult(:,1:2) = round(((sampleResult(:,1:2)-1)./8)+1); % Downsample grasp locations
        sampleResult(:,3) = mod(sampleResult(:,3),8); % Parallel-jaw grasp angles are equivalent in 180 degrees
    end

    % Load manual grasp labels
    try
        goodGraspPixLabels = dlmread(fullfile(labelDir,sprintf('%s.good.txt',sampleName))); % x1,y1,x2,y2 format
        goodGraspPixLabels(:,1:2:3) = goodGraspPixLabels(:,1:2:3) - 10; % Remove offsets from extra padding
        goodGraspPixLabels(:,2:2:4) = goodGraspPixLabels(:,2:2:4) - 12;
    catch
        goodGraspPixLabels = [];
    end
    try
        badGraspPixLabels = dlmread(fullfile(labelDir,sprintf('%s.bad.txt',sampleName)));
        badGraspPixLabels(:,1:2:3) = badGraspPixLabels(:,1:2:3) - 10; % Remove offsets from extra padding
        badGraspPixLabels(:,2:2:4) = badGraspPixLabels(:,2:2:4) - 12;
    catch
        badGraspPixLabels = [];
    end
    
    % Compare predictions against manual ground truth annotations
    for goodGraspIdx = 1:size(goodGraspPixLabels,1)
        goodGraspSample = goodGraspPixLabels(goodGraspIdx,:);
        goodGraspCenter = mean([goodGraspSample(1:2);goodGraspSample(3:4)]);
        goodGraspCenter = round((goodGraspCenter-1)./8+1); % Downsample grasp locations
        
        % Compute grasping direction and angle w.r.t. heightmap
        graspDirection = (goodGraspSample(1:2)-goodGraspSample(3:4))./norm((goodGraspSample(1:2)-goodGraspSample(3:4)));
        diffAngle = atan2d(graspDirection(1)*0-graspDirection(2)*1,graspDirection(1)*1+graspDirection(2)*0); % angle to 1,0
        while diffAngle < 0
            diffAngle = diffAngle+360;
        end
        rotIdx = mod(round(diffAngle/(45/2)),8);
        
        if ~isempty(sampleResult) && ismember([goodGraspCenter,rotIdx],sampleResult(:,1:3),'rows')
            sumTP = sumTP + 1;
        else
            sumFN = sumFN + 1;
        end
    end
    for badGraspIdx = 1:size(badGraspPixLabels,1)
        badGraspSample = badGraspPixLabels(badGraspIdx,:);
        badGraspCenter = mean([badGraspSample(1:2);badGraspSample(3:4)]);
        badGraspCenter = round((badGraspCenter-1)./8+1); % Downsample grasp locations
        
        % Compute grasping direction and angle w.r.t. heightmap
        graspDirection = (badGraspSample(1:2)-badGraspSample(3:4))./norm((badGraspSample(1:2)-badGraspSample(3:4)));
        diffAngle = atan2d(graspDirection(1)*0-graspDirection(2)*1,graspDirection(1)*1+graspDirection(2)*0); % angle to 1,0
        while diffAngle < 0
            diffAngle = diffAngle+360;
        end
        rotIdx = mod(round(diffAngle/(45/2)),8);
        
        if ~isempty(sampleResult) && ismember([badGraspCenter,rotIdx],sampleResult(:,1:3),'rows')
            sumFP = sumFP + 1;
        else
            sumTN = sumTN + 1;
        end
    end
end

% Compute total pixel-wise precision/recall over all test samples
precision = sumTP/(sumTP + sumFP);
recall = sumTP/(sumTP + sumFN);
fprintf('Precision: %f\nRecall: %f\n',precision,recall);








