# Introduction

This repository contains implementations for the major components of robot perception as part of MIT-Princeton's 1st Place winning entry (stow task) at the [Amazon Robotics Challenge](https://www.amazonrobotics.com/#/roboticschallenge) 2017. Featuring:

* [Multi-Affordance Grasping]() - a Torch implementation of fully convolutional neural networks for predicting multiple affordances from RGB-D images to detect suction-based grasps and parallel-jaw grasps for robotic picking.
* [Cross-Domain Image Matching]() - a Torch implementation of two-stream convolutional neural networks for matching observed images of grasped objects to their product images for recognition.
* [Baseline Algorithms]() - Matlab implementations of baselines for detecting grasps directly from 3D point cloud data.
* [Multi-Cam Realsense RGB-D Capture]() - A C++ implementation for streaming and capturing data (RGB-D frames) in real-time using [librealsense](https://github.com/IntelRealSense/librealsense), optimized to avoid IR depth interference for multi-cam setups. Tested on Ubuntu 16.04 connected to 16 Intel® RealSense™ SR300 Cameras.

<div align="center">
<img src="https://github.com/andyzeng/arc-robot-vision/raw/master/images/robot.jpg" height="250px" width="280px">
<img src="https://github.com/andyzeng/arc-robot-vision/raw/master/images/grasping.gif" height="250px" width="280px">
<img src="https://github.com/andyzeng/arc-robot-vision/raw/master/images/recognition.jpg" height="250px" width="280px">
</div>

![robot](images/robot.jpg?raw=true)![grasping](images/grasping.gif?raw=true)![recognition](images/recognition.jpg?raw=true)

For more information about our approach, please visit our [project webpage](http://apc.cs.princeton.edu/) and check out our [paper]():

### Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching ([pdf]() | [arxiv]() | [webpage]())

*[Andy Zeng](http://andyzeng.com), [Shuran Song](http://vision.princeton.edu/people/shurans/), [Kuan-Ting Yu](http://people.csail.mit.edu/peterkty/), [Elliott Donlon](https://www.linkedin.com/in/elliott-donlon-238601a3), [Francois R. Hogan](https://www.linkedin.com/in/francois-hogan-2b4025b6/), [Maria Bauza](http://web.mit.edu/bauza/www/), Daolin Ma, Orion Taylor, [Melody Liu](https://melodygl.wordpress.com/), Eudald Romo, [Nima Fazeli](http://nfazeli.mit.edu/), [Ferran Alet](http://web.mit.edu/alet/www/), [Nikhil Chavan Dafle](https://nikhilcd.mit.edu/), [Rachel Holladay](http://people.csail.mit.edu/rholladay/), Isabella Morona, [Prem Qu Nair](http://premqunair.com/), Druck Green, Ian Taylor, Weber Liu, [Thomas Funkhouser](http://www.cs.princeton.edu/~funk/), [Alberto Rodriguez](http://meche.mit.edu/people/faculty/ALBERTOR@MIT.EDU)*

**Abstract** This paper presents a robotic pick-and-place system that is capable of grasping and recognizing both known and novel objects in cluttered environments. The key new feature of the system is that it handles a wide range of object categories without needing any task-specific training data for novel objects. To achieve this, it first uses a category-agnostic affordance prediction algorithm to select among four different grasping primitive behaviors. It then recognizes picked objects with a cross-domain image classification framework that matches observed images to product images. Since product images are readily available for a wide range of objects (e.g., from the web), the system works out-of-the-box for novel objects without requiring any additional training data. Exhaustive experimental results demonstrate that our multi-affordance grasping achieves high success rates for a wide variety of objects in clutter, and our recognition algorithm achieves high accuracy for both known and novel grasped objects. The approach was part of the MIT-Princeton Team system that took 1st place in the stowing task at 2017 [Amazon Robotics Challenge](https://www.amazonrobotics.com/#/roboticschallenge).

#### Citing

If you find this code useful in your work, please consider citing:

```bash
@article{zeng2017robotic, 
	title={Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching}, 
	author={Zeng, Andy and Song, Shuran and Yu, Kuan-Ting and Donlon, Elliott and Hogan, Francois Robert and Bauza, Maria and Ma, Daolin and Taylor, Orion and Liu, Melody and Romo, Eudald and Fazeli, Nima and Alet, Ferran and Dafle, Nikhil Chavan and Holladay, Rachel and Morona, Isabella and Nair, Prem Qu and Green, Druck and Taylor, Ian and Liu, Weber and Funkhouser, Thomas and Rodriguez, Alberto}, 
	booktitle={arXiv preprint arXiv:xxx}, 
	year={2017} 
}
```

#### License
This code is released under the Apache License v2.0 (refer to the LICENSE file for details).

#### Datasets
Information and download links for our grasping dataset and image matching dataset can be found on our [project webpage](http://arc.cs.princeton.edu/).

#### Contact
If you have any questions or find any bugs, please let me know: [Andy Zeng](http://www.cs.princeton.edu/~andyz/) andyz[at]princeton[dot]edu

# Requirements and Dependencies

* NVIDIA GPU with compute capability 3.5+
* Torch with packages: image, optim, inn, cutorch, cunn, cudnn, hdf5
* Matlab 2015b or later

Our implementations have been tested on Ubuntu 16.04.

# Suction-Based Grasping

**Input**: An RGB-D image of various objects in a scene

**Output**: Confidence map of pixel-level affordances (where higher values indicate better locations for grasping with suction)



## Quick Start

1. Clone this repository and navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    git clone https://github.com/andyzeng/arc-robot-vision.git
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. Download our pre-trained model for suction-based grasping

    ```bash
    wget http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7
    ```

    Direct download link: [suction-based-grasping-snapshot-10001.t7 (450.1 MB)](http://vision.princeton.edu/projects/2017/arc/downloads/suction-based-grasping-snapshot-10001.t7)

3. Run our model on an optional target RGB-D image. Input color images should be 24-bit RGB PNG, while depth images should be 16-bit PNG, where depth values are saved in deci-millimeters (10<sup>-4</sup>m).

    ```bash
    th infer.lua # creates a results.h5 file
    ```

    or

    ```bash
    imgColorPath=<image.png> imgDepthPath=<image.png> modelPath=<model.t7> th infer.lua # creates a results.h5 file
    ```

4. Visualize the predictions in Matlab. Shows a heat map of confidence values where hotter regions indicate better locations for grasping with suction. Run the following in Matlab:


## Training

1. Navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    cd arc-robot-vision/suction-based-grasping
    ```

2. Download our suction-based grasping dataset. More information about the dataset can be found [here]().

    ```bash
    wget 
    unzip
    ```

 Direct download link: [suction-based-grasping-dataset.zip (1.6 GB)]()

 3. Run training (set optional parameters through command line arguments)

    ```bash
    cd convnet
    th train.lua
     ```

## Evaluation

1. Navigate to `arc-robot-vision/suction-based-grasping/convnet`

    ```bash
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. 

    ```bash
    cd arc-robot-vision/suction-based-grasping/convnet
    ```

2. Run our model on the testing split of our dataset

## Run Our Baseline Demo



# Parallel-Jaw Grasping













# Cross-Domain Image Matching




## Evaluation