% Script for post-processing and visualizing suction-based grasping
% affordance predictions

% User options (change me)
backgroundColorImage = 'demo/test-background.color.png';   % 24-bit RGB PNG
backgroundDepthImage = 'demo/test-background.depth.png';   % 16-bit PNG depth in deci-millimeters
inputColorImage = 'demo/test-image.color.png';             % 24-bit RGB PNG
inputDepthImage = 'demo/test-image.depth.png';             % 16-bit PNG depth in deci-millimeters
cameraIntrinsicsFile = 'demo/test-camera-intrinsics.txt';  % 3x3 camera intrinsics matrix
resultsFile = 'demo/results.h5';                           % HDF5 ConvNet output file from running infer.lua

% Read RGB-D images and intrinsics
backgroundColor = double(imread(backgroundColorImage))./255;
backgroundDepth = double(imread(backgroundDepthImage))./10000;
inputColor = double(imread(inputColorImage))./255;
inputDepth = double(imread(inputDepthImage))./10000;
cameraIntrinsics = dlmread(cameraIntrinsicsFile);

% Read raw affordance predictions
results = hdf5read(resultsFile,'results');
results = permute(results,[2,1,3,4]); % Flip x and y axes
affordanceMap = results(:,:,2); % 2nd channel contains positive affordance
affordanceMap = imresize(affordanceMap,size(inputDepth)); % Resize output to full  image size 

% Clamp affordances back to range [0,1] (after interpolation from resizing)
affordanceMap(affordanceMap >= 1) = 0.9999;
affordanceMap(affordanceMap < 0) = 0;

% Post-process affordance predictions and generate surface normals
[affordanceMap,surfaceNormalsMap] = postprocess(affordanceMap, ...
                                    inputColor,inputDepth, ...
                                    backgroundColor,backgroundDepth, ...
                                    cameraIntrinsics);

% Gaussian smooth affordances
affordanceMap = imgaussfilt(affordanceMap, 7);

% Generate heat map visualization for affordances
cmap = jet;
affordanceMap = cmap(floor(affordanceMap(:).*size(cmap,1))+1,:);
affordanceMap = reshape(affordanceMap,size(inputColor));

% Overlay affordance heat map over color image and save to results.png
figure(1); imshow(0.5*inputColor+0.5*affordanceMap);
figure(2); imshow(surfaceNormalsMap);
imwrite(0.5*inputColor+0.5*affordanceMap,'results.png')
imwrite(surfaceNormalsMap,'normals.png')
