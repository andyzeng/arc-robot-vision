function affordanceMap = postprocess(affordanceMap,inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics)
% Post-process affordance maps with background subtraction and removing
% regions with high variance in 3D surface normals
%
% function affordanceMap = postprocess(affordanceMap,inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics)
% Input:
%   affordanceMap     - 480x640 float array of affordance values in range [0,1]
%   inputColor        - 480x640x3 array of RGB color values scaled to range [0 - 255]
%   inputDepth        - 480x640 float array of depth values in meters
%   backgroundColor   - 480x640x3 array of RGB color values scaled to range [0 - 255]
%   backgroundDepth   - 480x640 float array of depth values in meters
%   cameraIntrinsics  - 3x3 camera intrinsics matrix
% Output:
%   affordanceMap     - 480x640 float array of post-processed affordance values in range [0,1]

% Perform background subtraction to get foreground mask
foregroundMaskColor = ~(sum(abs(inputColor-backgroundColor) < 0.3,3) == 3);
foregroundMaskDepth = backgroundDepth ~= 0 & abs(inputDepth-backgroundDepth) > 0.02;
foregroundMask = (foregroundMaskColor | foregroundMaskDepth);

% Project depth into 3D camera space
[pixX,pixY] = meshgrid(1:640,1:480);
camX = (pixX-cameraIntrinsics(1,3)).*inputDepth/cameraIntrinsics(1,1);
camY = (pixY-cameraIntrinsics(2,3)).*inputDepth/cameraIntrinsics(2,2);
camZ = inputDepth;
validDepth = foregroundMask & camZ ~= 0; % only points with valid depth and within foreground mask
inputPoints = [camX(validDepth),camY(validDepth),camZ(validDepth)]';

% Compute foreground point cloud normals
foregroundPointcloud = pointCloud(inputPoints');
foregroundNormals = pcnormals(foregroundPointcloud,50);

% Flip normals to point towards cameras
sensorCenter = [0,0,0];
for k = 1 : size(inputPoints,2)
   p1 = sensorCenter - [inputPoints(1,k),inputPoints(2,k),inputPoints(3,k)];
   p2 = [foregroundNormals(k,1),foregroundNormals(k,2),foregroundNormals(k,3)];
   angle = atan2(norm(cross(p1,p2)),p1*p2');
   if angle > pi/2 || angle < -pi/2
   else
       foregroundNormals(k,:) = -foregroundNormals(k,:);
   end
end

% Project normals back onto image plane
pixX = round(inputPoints(1,:)*cameraIntrinsics(1,1)./inputPoints(3,:)+cameraIntrinsics(1,3));
pixY = round(inputPoints(2,:)*cameraIntrinsics(2,2)./inputPoints(3,:)+cameraIntrinsics(2,3));
foregroundNormalsMap = zeros(size(inputColor));
foregroundNormalsMap(sub2ind(size(foregroundNormalsMap),pixY,pixX,ones(size(pixY)))) = foregroundNormals(:,1);
foregroundNormalsMap(sub2ind(size(foregroundNormalsMap),pixY,pixX,2*ones(size(pixY)))) = foregroundNormals(:,2);
foregroundNormalsMap(sub2ind(size(foregroundNormalsMap),pixY,pixX,3*ones(size(pixY)))) = foregroundNormals(:,3);

% Compute standard deviation of local normals
meanStdNormals = mean(stdfilt(foregroundNormalsMap,ones(25,25)),3);
normalBasedSuctionScores = 1 - meanStdNormals./max(meanStdNormals(:));

% Set affordance to 0 for regions with high surface normal variance
affordanceMap(normalBasedSuctionScores < 0.1) = 0;
affordanceMap(~foregroundMask) = 0;

end

