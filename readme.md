## Real-Time 3D Objects Detection and Segmentation based on Automotive LiDAR Technology   
### UPC - Universitat Politècnica de Catalunya  
### Postgraduate Course in Artificial Intelligence with Deep Learning  

---

![title_page_graphic](/F1_Documentation/docs/imgs/title_page.JPG)

---

| Author          | GitHub profile |
| ---           | --- |
|Pawel Dymek    | [pdymek](https://github.com/pdymek)|
|Nikolai Pchelin| [Niko1707](https://github.com/Niko1707) |
|Nil Oller      | [NilOller](https://github.com/NilOller)|
|Francesc Fons  | [ffons9](https://github.com/ffons9)|

---

## PROJECT OUTLINE

- [1.0. EXECUTIVE SUMMARY](#10-executive-summary)
- [2.0. AUTOMOTIVE AUTONOMOUS DRIVING DATASET](#20-automotive-autonomous-driving-dataset)
    - [2.1. 3D POINT CLOUD LiDAR SENSOR](#21-3d-point-cloud-lidar-sensor)
- [3.0. SEGMENTATION NEURAL NETWORK MODEL](#30-segmentation-neural-network-model)
- [4.0. DESIGN AND DEVELOPMENT](#40-design-and-development)
    - [4.1. SYSTEM ARCHITECTURE](#41-system-architecture)
    - [4.2. SOFTWARE CODING: TRAINING AND INFERENCE](#42-software-coding-training-and-inference)
    - [4.3 HARDWARE SETUP: PROCESSING PLATFORM](#43-hardware-setup-processing-platform)
- [5.0. EXPERIMENTAL RESULTS](#50-experimental-results)
    - [5.1. TRAINING PROCESS](#51-training-process)
    - [5.2. GROUND TRUTH vs PREDICTIONS](#52-ground-truth-vs-predictions)
- [6.0. CONCLUSIONS](#60-conclusions)
- [7.0. PROJECT RUN INSTRUCTIONS](#70-project-run-instructions)
    - [7.1. PREPARE KITTI DATASET CATALOG](#71-prepare-kitti-dataset-catalog)
    - [7.2. BUILD THE VIRTUAL ENVIRONMENT](#72-build-the-virtual-environment)
    - [7.3. TRAIN](#73-train)
    - [7.4. TEST](#74-test)
    - [7.5. VISUALIZATION](#75-visualization)


- [REFERENCES](#references)
---

## 1.0. EXECUTIVE SUMMARY

Hands on project consisting in the implementation of deeep learning (DL) techniques applied to the 3D objects detection and segmentation based on LiDAR technology in real autonomous driving scenarios.

Deployment of DL application targeting automotive autonomous driving / advanced driving assistance systems (AD/ADAS) by combining the following technologies and artifacts:

  - Neural Network Model: PointNet
  - Point Cloud Sensing: Velodyne HDL-64E LiDAR
  - Dataset: KITTI / SemanticKITTI
  - Point Cloud Viewer Tool: semantic-kitti-api
  - Computation Platform: Desktop PC equipped with CPU and GPU

The goal of this project was to put in practice the knowledge acquired along the postgraduate course by understanding and digesting the full development cycle of DL applications.

---

 
## 2.0. AUTOMOTIVE AUTONOMOUS DRIVING DATASET

The dataset selected in this project has been KITTI / SemanticKITTI.

https://www.cvlibs.net/datasets/kitti/  
http://www.semantic-kitti.org/  

--- 

### 2.1. 3D POINT CLOUD LiDAR SENSOR

The 3D point cloud data is acquired through the Velodyne HDL-64E LiDAR Sensor.  
The point cloud data format is depicted next.

![point_cloud_format](/F1_Documentation/docs/imgs/LiDAR_PointCloud.png)

--- 
## 3.0. SEGMENTATION NEURAL NETWORK MODEL

The Neural Network Model in use is PointNet.


![point_net_architecture_graphic](/F1_Documentation/docs/imgs/point_net_architecture.png)

---

## 4.0. DESIGN AND DEVELOPMENT

The main challenge of this project has been the fact of adapting and connecting many different pieces together: autonomous driving dataset, 3D LiDAR point clouds, segmentation NN model, visualization tools, etc. and programming all the application in python and pytorch programming language.

<!-- ![puzzle](/F1_Documentation/docs/imgs/PuzzleTechnologies.png) -->

<p align="center"><img src="/F1_Documentation/docs/imgs/PuzzleTechnologies.png" width="350" height="330"></p>

---

### 4.1. SYSTEM ARCHITECTURE

The software project has been architected to make our solution modular, flexible and scalable. To this aim, the full application is decomposed in building blocks that are easily interconnected giving place to a simple processing flow, as illustrated below.

![system_architecture_graphic](/F1_Documentation/docs/imgs/system_architecture.jpg)

---

The short period of time assigned to this project has forced the authors to teamwork following agile methodologies in order to iterate the final product in short sprints. The fact of having a modular architecture enabled each developer to focus on one specific building block with reasonable freedom of interference.

<!-- ![agile](/F1_Documentation/docs/imgs/Agile.png) -->

<p align="center"><img src="/F1_Documentation/docs/imgs/Agile.png" width="500" height="250"></p>

---

### 4.2. SOFTWARE CODING: TRAINING AND INFERENCE

The NN processing is split in two phases: training and inference.  
The outcome of the training (and validation) phase is a model stored in the way of a .pth file format.  
This .pth file is used later in the second phase related to inference (test) to perform the predictions.

<p align="center"><img src="/F1_Documentation/docs/gifs/From_Training_to_Inference_gif.gif" width="800" height="300"></p>

---

### 4.3 HARDWARE SETUP: PROCESSING PLATFORM

The project has been executed in a desktop PC consisting of one CPU and one GPU. DEVELOP  
The training is performed in the GPU whereas the test can run in the CPU.

<p align="center"><img src="/F1_Documentation/docs/imgs/HWsetup.png" width="800" height="450"></p>

---

## 5.0. EXPERIMENTAL RESULTS

Two different variants of the dataset have been used for training: (i) the original KITTI/SemanticKITTI dataset consisting of 22 different sequences of around 4500 time steps each and (ii) a lightweight version of KITTI/SemanticKITTI with less objects and a very reduced size of points.

--- 

### 5.1. TRAINING PROCESS

The full training process can be monitored through TensorBoard.

<!-- <img src="/F1_Documentation/docs/imgs/tensorboard1.JPG" width="500" height="250"> -->

![tensorboard1](/F1_Documentation/docs/imgs/tensorboard1.JPG)

---

### 5.2. GROUND TRUTH vs PREDICTIONS

The best way to assess our results is by having a look at the point cloud, as shown below.

Ground truth:  

![](/F1_Documentation/docs/gifs/Ground%20truth.gif)


Prediction results:  

![](/F1_Documentation/docs/gifs/Training%20result.gif)


---

## 6.0. CONCLUSIONS

Despite the short period of time devoted to the development of this project, authors could complete the full design cycle to reach some results that we consider are good enough according the objectives of this one-semester course.

All in all, this hands on exercise has been a good learning session in order for the authors to better understand and digest all the concepts and knowledge on Deep Learning presented along the course.

Great team and good job!

---

## 7.0. PROJECT RUN INSTRUCTIONS

---

### 7.1. PREPARE KITTI DATASET CATALOG

Link for dataset: http://www.semantic-kitti.org/dataset.html

Data for each of the scences should be organised with following structure. The mnimial rquired version is having one directory with velodyne data and one with labels.

```
  .../<directory_name>/sequences/
                                /<scence_number>/velodyne
                                                /labels
                                /<scence_number>/velodyne
                                                /labels                                                
                                ...
```
---

### 7.2. BUILD THE VIRTUAL ENVIRONMENT

To build the virtual environment from the terminal. For instance, you can do it manually from the powershell in Visual Studio Code by running the following instructions:

```
    python -m venv virtualenv
    virtualenv\Scripts\activate.bat
    pip install -r requirements.txt
```


---

### 7.3. TRAIN

Run the program through the following command from the terminal.

For training there are two required paramters only: `--hparamDatasetPath` and `--hparamActionType` (`=train`). The others would be taken as default from `hyperparam.py` file. However they also could be customized.  

Minimal example (with usage of default hyper parameters):

> python main.py --hparamActionType train --hparamDatasetPath G:\\Project\\_kitti_test\\data_odometry_velodyne\\dataset\\sequences\\

More evaluated example
> python main.py --hparamActionType train --hparamDatasetPath G:\\Project\\_kitti_test\\data_odometry_velodyne\\dataset\\sequences\\ --hparamDatasetSequence 00 --hparamDatasetSequence 04
> 
The other parameters that could be used in train:
    - `--hparamDatasetSequence` - number of sequence used for training, default '00'
    - `--hparamValDatasetSequence`- number of sequence used for validation, default '00'
    - `--hparamNumPoints` - default 4000, number of points for each of the scenes used in training
    - `--hparamTrainBatchSize` - training batch size
    - `--hparamValBatchSize` - validation batch size
    - `--hparamYamlConfigPath` - in case of use external .yaml config file
    - `--hparamPointDimension` - 3 or 4 for Kitty dataset (4 for the inclusion of reflectance)
    - `--hparamNumberOfClasses` - number of classes
    - `--hparamTrainNumEpochs`
    - `--hparamValNumEpochs`

---

### 7.4. TEST


During evaluation for the selected scene would be created folder with predicted labels (in `<seqence_number>/predictions` catalog).  

```
  .../<directory_name>/sequences/
                                /<scence_number>/velodyne
                                                /labels
                                                /predictions
                                ...
```

For test there are four required paramters only: `--hparamDatasetPath`,`--hparamActionType` (`=test`) and `--hparamModelPthPath` and `--hparamTestDatasetSequence`. The others would be taken as default from `hyperparam.py` file. However they also could be customized.  

Minimal example (with usage of default hyper parameters):

>python main.py --hparamActionType test --hparamDatasetPath G:\\Project\\_kitti_test\\data_odometry_velodyne\\dataset\\sequences\\ --hparamModelPthPath G:\\Project\\_kitti_test\\seg_model_bus_99.pth --hparamTestDatasetSequence 11

---

### 7.5. VISUALIZATION

Execute the following commands from the project root in the terminal to visualize both point cloud and voxels:

> python main.py --hparamDatasetPath E:\Project\Lidar_KITI\kitti\dataset --hparamDatasetSequence 11 --hparamPredictionsPath E:\Project\Lidar_KITI\kitti\dataset --hparamActionType visualize


Used parameters
- `--hparamActionType` - `"visualize"`
- `--hparamDatasetPath` - path for root dataset catalog
- `--hparamDatasetSequence` - number of visualized sequence 
- `--hparamPredictionsPath` - path for a directory with predictions


---

## REFERENCES

Along the different stages of our project, we have been inspired by previous related works available in other repositories and websites.
They have been useful material that provided us many insights about theoretical approaches, problem-solving, and also for some code.

Neural Network:
- https://github.com/Yvanali/KITTISeg
- https://github.com/fxia22/pointnet.pytorch
- https://github.com/marionacaros/3D-object-segmentation-light-PointNet

Point Cloud Visualization Tool:
- https://github.com/PRBonn/semantic-kitti-api

