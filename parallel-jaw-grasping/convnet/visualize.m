% Script for visualizing parallel-jaw grasping affordance predictions

% User options (change me)
inputColorHeightmap = 'demo/test-heightmap.color.png';  % 24-bit RGB PNG
rotationAngle = 360-(45/2)*5;                           % Rotation of heightmap
resultsFile = 'demo/results.h5';                        % HDF5 ConvNet output file from running infer.lua

% Read heightmap color image
color = double(imread(inputColorHeightmap))./255;

% Get mask to remove affordance predictions outside heightmap
validMask = zeros(40,40);
validMask(8:33,2:39) = 1;
validMask = imrotate(validMask,rotationAngle,'crop');

% Read raw affordance predictions
results = hdf5read(resultsFile,'results');
results = permute(results,[2,1,3,4]); % Flip x and y axes
affordanceMap = results(:,:,2); % 2nd channel contains positive affordance
affordanceMap(validMask == 0) = 0; % Apply mask
affordanceMap = imresize(affordanceMap,8); % Resize output to full  image size

% Clamp affordances back to range [0,1] (after interpolation from resizing)
affordanceMap(affordanceMap >= 1) = 0.9999999;
affordanceMap(affordanceMap < 0) = 0;

% Gaussian smooth affordances
affordanceMap = imgaussfilt(affordanceMap, 7);

% Generate heat map visualization for affordances
cmap = jet;
affordanceMap = cmap(floor(affordanceMap(:).*size(cmap,1))+1,:);
affordanceMap = reshape(affordanceMap,[320,320,3]);

% Overlay affordance heat map over color image and save to results.png
figure(1); imshow(0.5*color+0.5*affordanceMap)
imwrite(0.5*color+0.5*affordanceMap,'results.png')
