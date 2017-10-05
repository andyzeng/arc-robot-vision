# Robotic Pick-and-Place of Novel Objects in Clutter

This repository contains implementations for the major components of robot perception as part of MIT-Princeton's 1st Place winning entry (stow task) at the [Amazon Robotics Challenge](https://www.amazonrobotics.com/#/roboticschallenge) 2017. Featuring:

* [Suction-Based Grasping](#suction-based-grasping) - a Torch implementation of fully convolutional neural networks (FCNs) for directly predicting suction-based grasping affordances from RGB-D images.
    * [Baseline Algorithm](#baseline-algorithm) - a Matlab implementation of a baseline algorithm that predicts suction-based grasping affordances by computing the variance of surface normals of a 3D point cloud (projected from RGB-D images), where lower variance = higher affordance.
* [Parallel-Jaw Grasping](#parallel-jaw-grasping) - a Torch implementation of fully convolutional neural networks (FCNs) for directly predicting parallel-jaw grasping affordances from heightmaps (created from RGB-D images).
    * [Baseline Algorithm](#baseline-algorithm-1) - a Matlab implementation of a baseline algorithm for detecting anti-podal parallel-jaw grasps by detecting "hill-like" geometric stuctures over a 3D point cloud (projected from RGB-D images).
* [Image Matching](#image-matching) - a Torch implementation of two-stream convolutional neural networks for matching observed images of grasped objects to their product images for recognition.

<div align="center">
<img src="https://github.com/andyzeng/arc-robot-vision/raw/master/images/robot.jpg" height="230px" width="307px">
<img src="https://github.com/andyzeng/arc-robot-vision/raw/master/images/grasping.gif" height="230px" width="258px">
<img src="https://github.com/andyzeng/arc-robot-vision/raw/master/images/recognition.jpg" height="230px" width="258px">
</div>

For more information about our approach, please visit our [project webpage](http://arc.cs.princeton.edu/) and check out our [paper](https://arxiv.org/pdf/1710.01330.pdf):

### Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching ( [pdf](https://arxiv.org/pdf/1710.01330.pdf) | [arxiv](https://arxiv.org/abs/1710.01330) | [webpage](http://arc.cs.princeton.edu/) )

*[Andy Zeng](http://andyzeng.com), [Shuran Song](http://vision.princeton.edu/people/shurans/), [Kuan-Ting Yu](http://people.csail.mit.edu/peterkty/), [Elliott Donlon](https://www.linkedin.com/in/elliott-donlon-238601a3), [Francois R. Hogan](https://www.linkedin.com/in/francois-hogan-2b4025b6/), [Maria Bauza](http://web.mit.edu/bauza/www/), Daolin Ma, Orion Taylor, [Melody Liu](https://melodygl.wordpress.com/), Eudald Romo, [Nima Fazeli](http://nfazeli.mit.edu/), [Ferran Alet](http://web.mit.edu/alet/www/), [Nikhil Chavan Dafle](https://nikhilcd.mit.edu/), [Rachel Holladay](http://people.csail.mit.edu/rholladay/), Isabella Morona, [Prem Qu Nair](http://premqunair.com/), Druck Green, Ian Taylor, Weber Liu, [Thomas Funkhouser](http://www.cs.princeton.edu/~funk/), [Alberto Rodriguez](http://meche.mit.edu/people/faculty/ALBERTOR@MIT.EDU)*

**Abstract** This paper presents a robotic pick-and-place system that is capable of grasping and recognizing both known and novel objects in cluttered environments. The key new feature of the system is that it handles a wide range of object categories without needing any task-specific training data for novel objects. To achieve this, it first uses a category-agnostic affordance prediction algorithm to select among four different grasping primitive behaviors. It then recognizes picked objects with a cross-domain image classification framework that matches observed images to product images. Since product images are readily available for a wide range of objects (e.g., from the web), the system works out-of-the-box for novel objects without requiring any additional training data. Exhaustive experimental results demonstrate that our multi-affordance grasping achieves high success rates for a wide variety of objects in clutter, and our recognition algorithm achieves high accuracy for both known and novel grasped objects. The approach was part of the MIT-Princeton Team system that took 1st place in the stowing task at 2017 [Amazon Robotics Challenge](https://www.amazonrobotics.com/#/roboticschallenge).

#### Citing

If you find this code useful in your work, please consider citing:

```bash
@article{zeng2017robotic, 
	title={Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching}, 
	author={Zeng, Andy and Song, Shuran and Yu, Kuan-Ting and Donlon, Elliott and Hogan, Francois Robert and Bauza, Maria and Ma, Daolin and Taylor, Orion and Liu, Melody and Romo, Eudald and Fazeli, Nima and Alet, Ferran and Dafle, Nikhil Chavan and Holladay, Rachel and Morona, Isabella and Nair, Prem Qu and Green, Druck and Taylor, Ian and Liu, Weber and Funkhouser, Thomas and Rodriguez, Alberto}, 
	booktitle={arXiv preprint arXiv:1710.01330}, 
	year={2017} 
}
```

#### License
This code is released under the Apache License v2.0 (refer to the [LICENSE](LICENSE) file for details).

#### Datasets
Information and download links for our grasping dataset and image matching dataset can be found on our [project webpage](http://vision.princeton.edu/projects/2017/arc/#datasets).

#### Contact
If you have any questions or find any bugs, please let me know: [Andy Zeng](http://www.cs.princeton.edu/~andyz/) andyz[at]princeton[dot]edu

# Requirements and Dependencies

* NVIDIA GPU with compute capability 3.5+
* [Torch](http://torch.ch/) with packages: [image](https://github.com/torch/image), [optim](https://github.com/torch/optim), [inn](https://github.com/szagoruyko/imagine-nn), [cutorch](https://github.com/torch/cutorch), [cunn](https://github.com/torch/cunn), [cudnn](https://github.com/soumith/cudnn.torch), [hdf5](https://github.com/deepmind/torch-hdf5)
* [Matlab](https://www.mathworks.com/products/matlab.html) 2015b or later

Our implementations have been tested on Ubuntu 16.04 with an NVIDIA Titan X. Our full pick-and-place system implementation (outside the scope of this repository) uses a lightweight C++ ROS service as a wrapper to control Torch/Lua and Matlab processes via TCP sockets. Data is shared between the processes by reading and writing from RAMDisk.

# Suction-Based Grasping

A Torch implementation of fully convolutional neural networks for predicting pixel-level affordances (here higher values indicate better surface locations for grasping with suction) given an RGB-D image as input.

![suction-based-grasping](images/suction-based-grasping.jpg?raw=true)

## Quick Start

To run our pre-trained model to get pixel-level affordances for grasping with suction:

1. Clone this repository and navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    git clone https://github.com/andyzeng/arc-robot-vision.git
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. Download our pre-trained model for suction-based grasping:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7
    ```

    Direct download link: [suction-based-grasping-snapshot-10001.t7 (450.1 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7)

3. Run our model on an optional target RGB-D image. Input color images should be 24-bit RGB PNG, while depth images should be 16-bit PNG, where depth values are saved in deci-millimeters (10<sup>-4</sup>m).

    ```bash
    th infer.lua # creates results.h5
    ```

    or

    ```bash
    imgColorPath=<image.png> imgDepthPath=<image.png> modelPath=<model.t7> th infer.lua # creates results.h5
    ```

4. Visualize the predictions in Matlab. Shows a heat map of confidence values where hotter regions indicate better locations for grasping with suction. Also displays computed surface normals, which can be used to decide between robot motion primitives suction-down or suction-side. Run the following in Matlab:

    ```matlab
    visualize; % creates results.png and normals.png
    ```

## Training

To train your own model:

1. Navigate to `arc-robot-vision/suction-based-grasping`

    ```bash
    cd arc-robot-vision/suction-based-grasping
    ```

2. Download our suction-based grasping dataset and save the files into `arc-robot-vision/suction-based-grasping/data`. More information about the dataset can be found [here](http://vision.princeton.edu/projects/2017/arc/#datasets).

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip
    unzip suction-based-grasping-dataset.zip # unzip dataset
    ```

    Direct download link: [suction-based-grasping-dataset.zip (1.6 GB)](http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-dataset.zip)

3. Download the Torch ResNet-101 model pre-trained on ImageNet:

    ```bash
    cd convnet
    wget http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7
    ```

    Direct download link: [resnet-101.t7 (409.4 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7)

4. Run training (set optional parameters through command line arguments):

    ```bash
    th train.lua
     ```

    Tip: if you run out of GPU memory (CUDA error=2), reduce batch size or modify the network architecture in `model.lua` to use the smaller [ResNet-50 (256.7 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7) model pre-trained on ImageNet.

## Evaluation

To evaluate a trained model:

1. Navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. Run our pre-trained model to get affordance predictions for the testing split of our grasping dataset:

    ```bash
    th test.lua # creates evaluation-results.h5
    ```

    or run your own model:

    ```bash
    modelPath=<model.t7> th test.lua # creates evaluation-results.h5
    ```

3. Run the evaluation script in Matlab to compute pixel-level precision against manual annotations from the grasping dataset, as reported in our [paper](https://arxiv.org/pdf/1710.01330.pdf):

    ```matlab
    evaluate;
    ```

## Baseline Algorithm

Our baseline algorithm predicts affordances for suction-based grasping by first computing 3D surface normals of the point cloud (projected from the RGB-D image), then measuring the variance of the surface normals (higher variance = lower affordance). To run our baseline algorithm over the testing split of our grasping dataset:

1. Navigate to `arc-robot-vision/suction-based-grasping/baseline`

    ```bash
    cd arc-robot-vision/suction-based-grasping/baseline
    ```

2. Run the following in Matlab:

    ```matlab
    test; % creates results.mat
    evaluate;
    ```

# Parallel-Jaw Grasping

A Torch implementation of fully convolutional neural networks for predicting pixel-level affordances for parallel-jaw grasping. The network takes an RGB-D heightmap as input, and outputs affordances for **horizontal** grasps. Input heightmaps can be rotated at any arbitrary angle. This structure allows the use of a unified model to predict grasp affordances for any possible grasping angle. 

![parallel-jaw-grasping](images/parallel-jaw-grasping.jpg?raw=true)

**Heightmaps** are generated by orthographically re-projecting 3D point clouds (from RGB-D images) upwards along the gravity direction where the height value of bin bottom = 0 (see [getHeightmap.m](parallel-jaw-grasping/convnet/getHeightmap.m)).

## Quick Start

To run our pre-trained model to get pixel-level affordances for parallel-jaw grasping:

1. Clone this repository and navigate to `arc-robot-vision/parallel-jaw-grasping/convnet`

    ```bash
    git clone https://github.com/andyzeng/arc-robot-vision.git
    cd arc-robot-vision/parallel-jaw-grasping/convnet
    ```

2. Download our pre-trained model for parallel-jaw grasping:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-snapshot-20001.t7
    ```

    Direct download link: [parallel-jaw-grasping-snapshot-20001.t7 (450.1 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-snapshot-20001.t7)

3. To generate a RGB-D heightmap given two RGB-D images, run the following in Matlab:

    ```matlab
    getHeightmap;
    ```

4. Run our model on an optional target RGB-D heightmap. Input color images should be 24-bit RGB PNG, while height images (depth) should be 16-bit PNG, where height values are saved in deci-millimeters (10<sup>-4</sup>m) and bin bottom = 0.

    ```bash
    th infer.lua # creates results.h5
    ```

    or

    ```bash
    imgColorPath=<image.png> imgDepthPath=<image.png> modelPath=<model.t7> th infer.lua # creates results.h5
    ```

5. Visualize the predictions in Matlab. Shows a heat map of confidence values where hotter regions indicate better locations for horizontal parallel-jaw grasping. Run the following in Matlab:

    ```matlab
    visualize; % creates results.png
    ```

## Training

To train your own model:

1. Navigate to `arc-robot-vision/parallel-jaw-grasping`

    ```bash
    cd arc-robot-vision/parallel-jaw-grasping
    ```

2. Download our parallel-jaw grasping dataset and save the files into `arc-robot-vision/parallel-jaw-grasping/data`. More information about the dataset can be found [here](http://vision.princeton.edu/projects/2017/arc/#datasets).

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-dataset.zip
    unzip parallel-jaw-grasping-dataset.zip # unzip dataset
    ```

    Direct download link: [parallel-jaw-grasping-dataset.zip (711.8 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/parallel-jaw-grasping-dataset.zip)

3. Pre-process input data and labels for parallel-jaw grasping dataset and save the files into `arc-robot-vision/parallel-jaw-grasping/convnet/training`. Pre-processing includes rotating heightmaps into 16 discrete rotations, converting raw grasp labels (two-point lines) into dense pixel-wise labels, and augmenting labels with small amounts of jittering. Either run the following in Matlab:

    ```matlab
    cd convnet;
    processLabels;
    ```

    or download our already pre-processed input:

    ```bash
    cd convnet;
    wget http://vision.princeton.edu/projects/2017/arc/downloads/training-parallel-jaw-grasping-dataset.zip
    unzip training-parallel-jaw-grasping-dataset.zip # unzip dataset
    ```

    Direct download link: [training-parallel-jaw-grasping-dataset.zip (740.0 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/training-parallel-jaw-grasping-dataset.zip)

4. Download the Torch ResNet-101 model pre-trained on ImageNet:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7
    ```

    Direct download link: [resnet-101.t7 (409.4 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-101.t7)

5. Run training (set optional parameters through command line arguments):

    ```bash
    th train.lua
     ```

    Tip: if you run out of GPU memory (CUDA error=2), reduce batch size or modify the network architecture in `model.lua` to use the smaller [ResNet-50 (256.7 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7) model pre-trained on ImageNet.

## Evaluation

To evaluate a trained model:

1. Navigate to `arc-robot-vision/parallel-jaw-grasping/convnet`

    ```bash
    cd arc-robot-vision/parallel-jaw-grasping/convnet
    ```

2. Run the model to get affordance predictions for the testing split of our grasping dataset:

    ```bash
    modelPath=<model.t7> th test.lua # creates evaluation-results.h5
    ```

3. Run the evaluation script in Matlab to compute pixel-level precision against manual annotations from the grasping dataset, as reported in our [paper](https://arxiv.org/pdf/1710.01330.pdf):

    ```matlab
    evaluate;
    ```

## Baseline Algorithm

Our baseline algorithm detects anti-podal parallel-jaw grasps by detecting "hill-like" geometric features (through brute-force sliding window search) from the 3D point cloud of an input heightmap (no color). These geometric features should satisfy two constraints: (1) gripper fingers fit within the concavities along the sides of the hill, and (2) top of the hill should be at least 2cm above the lowest points of the concavities. A valid grasp is ranked by an affordance score, which is computed by the percentage of 3D surface points between the gripper fingers that are above the lowest points of the concavities. To run our baseline algorithm over the testing split of our grasping dataset:

1. Navigate to `arc-robot-vision/parallel-jaw-grasping/baseline`

    ```bash
    cd arc-robot-vision/parallel-jaw-grasping/baseline
    ```
    
2. Run the following in Matlab:

    ```matlab
    test; % creates results.mat
    evaluate;
    ```

# Image Matching

A Torch implementation of two-stream convolutional neural networks for matching observed images of grasped objects to their product images for recognition. One stream computes 2048-dimensional feature vectors for product images while the other stream computes 2048-dimensional feature vectors for observed images. During training, both streams are optimized so that features are more similar for images of the same object and dissimilar otherwise. During testing, product images of both known and novel objects are mapped onto a common feature space. We recognize observed images by mapping them to the same feature space and finding the nearest neighbor product image match.

![image-matching](images/image-matching.jpg?raw=true)

## Training

To train a model:

1. Navigate to `arc-robot-vision/image-matching`

    ```bash
    cd arc-robot-vision/image-matching
    ```

2. Download our image matching dataset and save the files into `arc-robot-vision/image-matching/data`. More information about the dataset can be found [here](http://vision.princeton.edu/projects/2017/arc/#datasets).

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/image-matching-dataset.zip
    unzip image-matching-dataset.zip # unzip dataset
    ```

    Direct download link: [image-matching-dataset.zip (4.6 GB)](http://vision.princeton.edu/projects/2017/arc/downloads/image-matching-dataset.zip)

3. Download the Torch ResNet-50 model pre-trained on ImageNet:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7
    ```

    Direct download link: [resnet-50.t7 (256.7 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/resnet-50.t7)

4. Run training (change variable `trainMode` in `train.lua` depending on which architecture you want to train):

    ```bash
    th train.lua
     ```

## Evaluation

To evaluate a trained model:

1. Navigate to `arc-robot-vision/image-matching`

    ```bash
    cd arc-robot-vision/image-matching
    ```

2. Download our pre-trained models (K-net and N-net) for two-stage cross-domain image matching:

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/k-net.zip
    unzip k-net.zip 
    wget http://vision.princeton.edu/projects/2017/arc/downloads/n-net.zip
    unzip n-net.zip 
    ```

    Direct download links: [k-net.zip  (175.3 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/k-net.zip) and [n-net.zip (174.0 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/n-net.zip)

3. Run our pre-trained models to compute features for the testing split of our image matching dataset (change variable `trainMode` depending on which architecture you want to test):

    ```bash
    trainMode=1 snapshotsFolder=snapshots-with-class snapshotName=snapshot-170000 th test.lua # for k-net: creates HDF5 output file and saves into snapshots folder
    trainMode=2 snapshotsFolder=snapshots-no-class snapshotName=snapshot-8000 th test.lua # for n-net: creates HDF5 output file and saves into snapshots folder
    ```

4. Run the evaluation script in Matlab to compute 1 vs 20 object recognition accuracies over our image matching dataset, as reported in our [paper](https://arxiv.org/pdf/1710.01330.pdf):

    ```matlab
    evaluateTwoStage;
    ```

    or run the following in Matlab for evaluation on a single model (instead of a two stage system):

    ```matlab
    evaluateModel;
    ```
