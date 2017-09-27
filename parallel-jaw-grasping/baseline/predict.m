function [graspPredictions,flushGraspPredictions] = predict(heightmap)
% A baseline algorithm for detecting anti-podal parallel-jaw grasps and
% flush grasps by detecting "hill-like" geometric features (through
% brute-force sliding window search) from the 3D point cloud of an input
% heightmap (no color). These geometric features should satisfy two
% constraints: (1) gripper fingers fit within the concavities along the
% sides of the hill, and (2) top of the hill should be at least 2cm above
% the lowest points of the concavities. A valid grasp is ranked by an
% affordance score, which is computed by the percentage of 3D surface
% points between the gripper fingers that are above the lowest points of
% the concavities. 
%
% function [graspPredictions,flushGraspPredictions] = predict(heightmap)
% Input:
%   heightmap  - 200x300x3 float array of height values (from bottom
%                of bin) in meters. By default, each pixel has a
%                width of 0.002m
% Output:
%   graspPredictions       - Nx4 float array [x,y,r,c] with center of grasp
%                            in heightmap pixel coordinates (x,y), rotation
%                            index between [0,7] (r) where the difference
%                            between consecutive values is 22.5 degrees,
%                            and the confidence score in range [0,1] (c)
%   flushGraspPredictions  - same representation as graspPredictions

% User options (change me)
voxelSize = 0.002;        % width of each pixel in meters (resolution of heightmap)
fingerWidth = 0.06;       % width of gripper finger in meters (orthogonal to jaw motion)
fingerThickness = 0.036;  % thickness of gripper finger in meters (parallel to jaw motion)

graspPredictions = [];
flushGraspPredictions = [];

% Define constants (for speed)
heightmapSize = size(heightmap);
graspRotations = 0:deg2rad(45/2):deg2rad(179);

% Define pixel grid for grasp fingers
[topFingerInitX,topFingerInitY] = meshgrid(1:(fingerWidth/voxelSize),(fingerThickness/voxelSize):-1:1);
[botFingerInitX,botFingerInitY] = meshgrid(1:(fingerWidth/voxelSize),1:(fingerThickness/voxelSize));

% Predict normal grasps (brute force matching)
for dx = 5:25
    for dy = 6:15
        localHeightMap = heightmap((dy*10-9):(dy*10),(dx*10-9):(dx*10));
        if sum(localHeightMap(:) > 0) > 75
            xCoord = dx*10-5;
            yCoord = dy*10-5;
            xyCoordRep = repmat([xCoord;yCoord],1,size(topFingerInitX(:),1));
            medianLocalHeightmap = median(localHeightMap(:));
            % surfaceCentroid = ; % [xCoord;yCoord] in world coordinates (depends on robot system)

            % Rotate grasp orientations (in radians)
            for theta = graspRotations
                rotMat = [cos(theta),-sin(theta);sin(theta),cos(theta)];
                topFingerShiftedX = topFingerInitX - (fingerWidth/voxelSize)/2;
                botFingerShiftedX = botFingerInitX - (fingerWidth/voxelSize)/2;
                
                % Explore different grasp widths between fingers (in meters)
                for graspWidth = 0.03:0.02:0.11
                    graspWidthPix = (graspWidth/voxelSize)/2;
                    topFingerShiftedY = topFingerInitY + graspWidthPix;
                    topFingerPix = [topFingerShiftedX(:),topFingerShiftedY(:)];
                    botFingerShiftedY = -botFingerInitY - graspWidthPix;
                    botFingerPix = [botFingerShiftedX(:),botFingerShiftedY(:)];
                    topFingerPix = round(rotMat * topFingerPix' + xyCoordRep)';
                    botFingerPix = round(rotMat * botFingerPix' + xyCoordRep)';
                    fingerInd = sub2ind2d(heightmapSize, [topFingerPix(:,2);botFingerPix(:,2)],[topFingerPix(:,1);botFingerPix(:,1)]);

                    % Skip current sample if grasp is outside heightmap bounds
                    if min(fingerInd) < 0 || max(fingerInd) > length(heightmap(:))
                        continue;
                    end

                    % If middle surface centroid is higher than finger lowpoints, set valid grasp
                    fingerIndHeightmap = heightmap(fingerInd);
                    if medianLocalHeightmap > median(fingerIndHeightmap) + 0.02 && ...  %% median 50% is O(n) so can filter quickly
                       medianLocalHeightmap > prctile(fingerIndHeightmap,90) + 0.02     %% prctile is O(nlogn) because of sorting
                        [midGraspX,midGraspY] = meshgrid(1:(fingerWidth/voxelSize),1:(graspWidth/voxelSize));
                        midGraspX = midGraspX - (fingerWidth/voxelSize)/2;
                        midGraspY = midGraspY - (graspWidth/voxelSize)/2;
                        midGraspPix = [midGraspX(:),midGraspY(:)];
                        midGraspPix = round(rotMat * midGraspPix' + repmat([xCoord;yCoord],1,size(midGraspPix,1)))';  
                        graspInd = sub2ind2d(size(heightmap),midGraspPix(:,2),midGraspPix(:,1));
                        surfacePts = heightmap(graspInd) > median(heightmap(fingerInd));
                        graspConf = max((sum(surfacePts(:))./size(graspInd,1)),0);
                        graspPtsPix = [topFingerPix(18*16,:),botFingerPix(18*15+1,:)];
                        
                        % Draw grasps
                        colorMapJet = jet;
                        colorScale = colorMapJet(floor(graspConf.*63)+1,:);
                        hold on; plot([graspPtsPix(1);graspPtsPix(3)],[graspPtsPix(2);graspPtsPix(4)],'LineWidth',2,'Color',colorScale);
                        
                        % Compute grasping center and angle w.r.t. heightmap
                        graspCenterPix = mean([graspPtsPix(1:2);graspPtsPix(3:4)]);
                        graspDirection = (graspPtsPix(1:2)-graspPtsPix(3:4))./norm((graspPtsPix(1:2)-graspPtsPix(3:4)));
                        diffAngle = atan2d(graspDirection(1)*0-graspDirection(2)*1,graspDirection(1)*1+graspDirection(2)*0); % angle to 1,0
                        while diffAngle < 0
                            diffAngle = diffAngle+360;
                        end
                        rotIdx = mod(round(diffAngle/(45/2)),8); % Parallel-jaw grasp angles are equivalent in 180 degrees
                        graspPredictions = [graspPredictions;[graspCenterPix,rotIdx,graspConf]];
                        
                        % Additional grasp parameterization (depends on motion primitives)
                        % graspDirection = [0,0,-1];
                        % graspDepth = min(0.15, medianLocalHeightmap - prctile(heightmap(fingerInd),10));
                        % graspJawWidth = min(0.11,graspWidth+0.03);
                        % gripperAngleDirection = [0,1,0];
                        % gripperAngleAxis = [0,0,1];
                        % gripperRotm = vrrotvec2mat([gripperAngleAxis,theta]);
                        % gripperAngleDirection = (gripperRotm*gripperAngleDirection')';
                        % graspPredictions = [graspPredictions;[surfaceCentroid,graspDirection,graspDepth,graspJawWidth,gripperAngleDirection,graspConf]];

                        break; % Only return valid grasp with smallest width
                    end
                end
            end
        end
    end
end

% Predict flush grasps (brute force matching)
for dx=5:25
    
    % Flush grasps on upper side of bin (w.r.t. heighmap)
    for dy = [4,5]
        localHeightMap = heightmap((dy*10-9):(dy*10),(dx*10-9):(dx*10));
        if sum(localHeightMap(:) > 0) > 75
            xCoord = dx*10-5;
            yCoord = dy*10-5;
            medianLocalHeightmap = median(localHeightMap(:));
            % surfaceCentroid = ; % [xCoord;yCoord] in world coordinates (depends on robot system)

            % Explore different grasp widths between finger and wall (in meters)
            for graspWidth = 0.03:0.02:0.11
                [botFingerShiftedX,botFingerShiftedY] = meshgrid((xCoord+1-fingerWidth/(voxelSize*2)):(xCoord+fingerWidth/(voxelSize*2)),(21+(graspWidth/voxelSize)):(20+(graspWidth/voxelSize)+(fingerThickness/voxelSize)));
                botFingerPix = [botFingerShiftedX(:),botFingerShiftedY(:)];
              
                fingerInd = sub2ind(size(heightmap),botFingerPix(:,2),botFingerPix(:,1));

                % Skip current sample if grasp is outside heightmap bounds
                if min(fingerInd) < 0 || max(fingerInd) > length(heightmap(:))
                    continue;
                end

                % If middle surface centroid is higher than finger lowpoints, set valid grasp
                if medianLocalHeightmap > prctile(heightmap(fingerInd),90) + 0.02
                    [midGraspX,midGraspY] = meshgrid((xCoord-fingerWidth/(voxelSize*2)):(xCoord+fingerWidth/(voxelSize*2)),20:(20+(graspWidth/voxelSize)));
                    midGraspPix = [midGraspX(:),midGraspY(:)];
                    graspInd = sub2ind(size(heightmap),midGraspPix(:,2),midGraspPix(:,1));
                    surfacePts = heightmap(graspInd) > median(heightmap(fingerInd));
                    graspConf = max((sum(surfacePts(:))./size(graspInd,1)),0);
                    graspPtsPix = [xCoord,20,botFingerPix(18*14+1,:)];
                    
                    % Draw grasps
                    colorMapJet = jet;
                    colorScale = colorMapJet(floor(graspConf.*63)+1,:);
                    hold on; plot([graspPtsPix(1);graspPtsPix(3)],[graspPtsPix(2);graspPtsPix(4)],'LineWidth',2,'Color',colorScale);
                    
                    % Compute grasping center and angle w.r.t. heightmap
                    graspCenterPix = mean([graspPtsPix(1:2);graspPtsPix(3:4)]);
                    graspDirection = (graspPtsPix(1:2)-graspPtsPix(3:4))./norm((graspPtsPix(1:2)-graspPtsPix(3:4)));
                    diffAngle = atan2d(graspDirection(1)*0-graspDirection(2)*1,graspDirection(1)*1+graspDirection(2)*0); % angle to 1,0
                    while diffAngle < 0
                        diffAngle = diffAngle+360;
                    end
                    rotIdx = mod(round(diffAngle/(45/2)),8); % Parallel-jaw grasp angles are equivalent in 180 degrees
                    flushGraspPredictions = [flushGraspPredictions;[graspCenterPix,rotIdx,graspConf]];
                    
                    % Additional grasp parameterization (depends on motion primitives)
                    % graspJawWidth = min(0.11,graspWidth+0.01);
                    % graspDepth = min(0.15, medianLocalHeightmap - prctile(heightmap(fingerInd),10));
                    % flushGraspPredictions = [flushGraspPredictions;[surfaceCentroid,graspDepth,graspJawWidth,graspConf]];
                    % flushGraspVis = [flushGraspVis;[xCoord,20,botFingerPix(256,1),botFingerPix(256,2),graspConf,colorScale]];

                    break; % Only return valid grasp with smallest width
                end
            end
        end
    end
    
    % Flush grasps on lower side of bin (w.r.t. heighmap)
    for dy = [16,17]
        localHeightMap = heightmap((dy*10-9):(dy*10),(dx*10-9):(dx*10));
        if sum(localHeightMap(:) > 0) > 75
            xCoord = dx*10-5;
            yCoord = dy*10-5;
            % surfaceCentroid = ; % [xCoord;yCoord] in world coordinates (depends on robot system)

            % Explore different grasp widths between finger and wall (in meters)
            for graspWidth = 0.03:0.02:0.11
                [midGraspX,midGraspY] = meshgrid((xCoord-fingerWidth/(voxelSize*2)):(xCoord+fingerWidth/(voxelSize*2)),(180-(graspWidth/voxelSize)):180);
                midGraspPix = [midGraspX(:),midGraspY(:)];
                [topFingerShiftedX,topFingerShiftedY] = meshgrid((xCoord+1-fingerWidth/(voxelSize*2)):(xCoord+fingerWidth/(voxelSize*2)),(180+1-(graspWidth/voxelSize)-(fingerThickness/voxelSize)):(180-(graspWidth/voxelSize)));
                topFingerPix = [topFingerShiftedX(:),topFingerShiftedY(:)];
                fingerInd = sub2ind(size(heightmap),topFingerPix(:,2),topFingerPix(:,1));
                graspInd = sub2ind(size(heightmap),midGraspPix(:,2),midGraspPix(:,1));

                % Skip current sample if grasp is outside heightmap bounds
                if min(fingerInd) < 0 || max(fingerInd) > length(heightmap(:))
                    continue;
                end

                % If middle surface centroid is higher than finger lowpoints, set valid grasp
                if heightmap(yCoord,xCoord) > prctile(heightmap(fingerInd),90) + 0.02
                    surfacePts = heightmap(graspInd) > median(heightmap(fingerInd));
                    graspConf = max((sum(surfacePts(:))./size(graspInd,1)),0);
                    graspPtsPix = [xCoord,180,topFingerPix(18*15,:)];
                    
                    % Draw grasps
                    colorMapJet = jet;
                    colorScale = colorMapJet(floor(graspConf.*63)+1,:);
                    hold on; plot([graspPtsPix(1);graspPtsPix(3)],[graspPtsPix(2);graspPtsPix(4)],'LineWidth',2,'Color',colorScale);

                    % Compute grasping center and angle w.r.t. heightmap
                    graspCenterPix = mean([graspPtsPix(1:2);graspPtsPix(3:4)]);
                    graspDirection = (graspPtsPix(1:2)-graspPtsPix(3:4))./norm((graspPtsPix(1:2)-graspPtsPix(3:4)));
                    diffAngle = atan2d(graspDirection(1)*0-graspDirection(2)*1,graspDirection(1)*1+graspDirection(2)*0); % angle to 1,0
                    while diffAngle < 0
                        diffAngle = diffAngle+360;
                    end
                    rotIdx = mod(round(diffAngle/(45/2)),8); % Parallel-jaw grasp angles are equivalent in 180 degrees
                    flushGraspPredictions = [flushGraspPredictions;[graspCenterPix,rotIdx,graspConf]];
                    
                    % Additional grasp parameterization (depends on motion primitives)
                    % graspJawWidth = min(0.11,graspWidth+0.01);
                    % graspDepth = min(0.15,heightmap(yCoord,xCoord) - prctile(heightmap(fingerInd),10));
                    % flushGraspPredictions = [flushGraspPredictions;[surfaceCentroid,graspDepth,graspJawWidth,graspConf]];
                    % flushGraspVis = [flushGraspVis;[xCoord,180,topFingerPix(180,1),topFingerPix(180,2),graspConf,colorScale]]; % FIX THIS LATER
                    
                    break; % Only return valid grasp with smallest width
                end
            end
        end
    end
end

end