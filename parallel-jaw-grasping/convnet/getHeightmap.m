% This is a demo version for generating a heightmap from two RGB-D images.
% Project RGB-D data into a 3D point cloud, then generate a heightmap by
% orthographically reprojecting the point cloud along the gravity
% direction.

% Read fixed position of bin w.r.t. world coordinates (assumes fixed orientation)
binMiddleBottom = dlmread('demo/bin-position.txt');

% Initialize variables
heightMaps = {};
missingHeightMaps = {};
voxelSize = 0.002; % Size of each heightmap pixel in world coordinates
heightMapColor = zeros(200*300,3);

% Use two RGB-D images (captured from two different cameras) to
% construct a unified height map
for camIdx = 0:1
    colorImgPath = sprintf('demo/input-%d.color.png',camIdx);
    depthImgPath = sprintf('demo/input-%d.depth.png',camIdx);
    bgColorImgPath = sprintf('demo/background-%d.color.png',camIdx);
    bgDepthImgPath = sprintf('demo/background-%d.depth.png',camIdx);
    camIntrinsicsPath = sprintf('demo/camera-%d.intrinsics.txt',camIdx);
    camPosePath = sprintf('demo/camera-%d.pose.txt',camIdx);

    % Read RGB-D image files
    colorImg = double(imread(colorImgPath))./255;
    depthImg = double(imread(depthImgPath))./10000;
    bgColorImg = double(imread(bgColorImgPath))./255;
    bgDepthImg = double(imread(bgDepthImgPath))./10000;
    camIntrinsics = dlmread(camIntrinsicsPath);
    camPose = dlmread(camPosePath);

    % Do background subtraction
    foregroundMaskColor = ~(sum(abs(colorImg-bgColorImg) < 0.3,3) == 3);
    foregroundMaskDepth = bgDepthImg ~= 0 & abs(depthImg-bgDepthImg) > 0.02;
    foregroundMask = (foregroundMaskColor | foregroundMaskDepth);

    % Project depth into camera space
    [pixX,pixY] = meshgrid(1:640,1:480);
    camX = (pixX-camIntrinsics(1,3)).*depthImg/camIntrinsics(1,1);
    camY = (pixY-camIntrinsics(2,3)).*depthImg/camIntrinsics(2,2);
    camZ = depthImg;
    camPts = [camX(:),camY(:),camZ(:)];

    % Transform points to world coordinates
    worldPts = (camPose(1:3,1:3)*camPts' + repmat(camPose(1:3,4),1,size(camPts,1)))';

    % Get height map
    heightMap = zeros(200,300);
    gridOrigin = [binMiddleBottom(1)-0.3,binMiddleBottom(2)-0.2,binMiddleBottom(3)];
    gridMapping = [round((worldPts(:,1)-gridOrigin(1))./voxelSize), round((worldPts(:,2)-gridOrigin(2))./voxelSize), worldPts(:,3) - binMiddleBottom(3)];

    % Compute height map color
    validPix = gridMapping(:,1) > 0 & gridMapping(:,1) <= 300 & gridMapping(:,2) > 0 & gridMapping(:,2) <= 200; %& gridMapping(:,3) > 0;
    colorPts = [reshape(colorImg(:,:,1),[],1),reshape(colorImg(:,:,2),[],1),reshape(colorImg(:,:,3),[],1)];
    heightMapColor(sub2ind(size(heightMap),gridMapping(validPix,2),gridMapping(validPix,1)),:) = colorPts(validPix,:);

    % Compute real height map with background subtraction
    validPix = gridMapping(:,1) > 0 & gridMapping(:,1) <= 300 & gridMapping(:,2) > 0 & gridMapping(:,2) <= 200& gridMapping(:,3) > 0;
    validDepth = (foregroundMask & camZ ~= 0);
    gridMapping = gridMapping(validPix&validDepth(:),:);
    heightMap(sub2ind(size(heightMap),gridMapping(:,2),gridMapping(:,1))) = gridMapping(:,3);

    % Find missing depth and project background depth into camera space
    missingDepth = depthImg == 0 & bgDepthImg > 0;
    [pixX,pixY] = meshgrid(1:640,1:480);
    camX = (pixX-camIntrinsics(1,3)).*bgDepthImg/camIntrinsics(1,1);
    camY = (pixY-camIntrinsics(2,3)).*bgDepthImg/camIntrinsics(2,2);
    camZ = bgDepthImg;
    missingCamPts = [camX(missingDepth),camY(missingDepth),camZ(missingDepth)];
    missingWorldPts = (camPose(1:3,1:3)*missingCamPts' + repmat(camPose(1:3,4),1,size(missingCamPts,1)))';

    % Get missing depth height map
    missingHeightMap = zeros(200,300);
    gridOrigin = [binMiddleBottom(1)-0.3,binMiddleBottom(2)-0.2,binMiddleBottom(3)];
    gridMapping = [round((missingWorldPts(:,1)-gridOrigin(1))./voxelSize), round((missingWorldPts(:,2)-gridOrigin(2))./voxelSize), missingWorldPts(:,3) - binMiddleBottom(3)];
    validPix = gridMapping(:,1) > 0 & gridMapping(:,1) <= 300 & gridMapping(:,2) > 0 & gridMapping(:,2) <= 200;
    gridMapping = gridMapping(validPix,:);
    missingHeightMap(sub2ind(size(missingHeightMap),gridMapping(:,2),gridMapping(:,1))) = 1;

    noisePix = ~bwareaopen(missingHeightMap > 0,50);
    missingHeightMap(noisePix) = 0;

    % Denoise height map
    noisePix = ~bwareaopen(heightMap > 0,50);
    heightMap(noisePix) = 0;

    heightMaps{camIdx+1} = heightMap;
    missingHeightMaps{camIdx+1} = missingHeightMap;
end
heightMap = max(heightMaps{1},heightMaps{2});
heightMapColor = reshape(heightMapColor,[200,300,3]);

% Height cannot exceed 30cm above bottom of tote
heightMap = min(heightMap,ones(size(heightMap)).*0.3);
rawHeightMap = heightMap;

% Fill in missing depth holes (assume height of 3cm)
heightMap(heightMap == 0 & (missingHeightMaps{1} & missingHeightMaps{2})) = 0.03;

% Flip heightmap y-axis
heightMapColor = flipud(heightMapColor);
rawHeightMap = flipud(rawHeightMap);

% Save height map and reprojected color images with extra padding
% Padding: +12px both side y-axis, +10px both side x-axis
colorData = zeros(224,320,3);
depthData = uint16(zeros(224,320));
colorData(13:212,11:310,:) = heightMapColor;
depthData(13:212,11:310) = uint16(rawHeightMap.*10000);
imwrite(colorData,'demo/raw-heightmap.color.png');
imwrite(depthData,'demo/raw-heightmap.depth.png');

