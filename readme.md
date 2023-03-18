


<!-- <img src="/F1_Documentation/docs/imgs/title_page.JPG" title="asfsf"/> -->


![title_page_graphic](/F1_Documentation/docs/imgs/title_page.JPG)

# Real-Time 3D Objects Detection and Segmentation based on Automotive LiDAR Technology

## Postgraduate course, UPC Universitat Politècnica de Catalunya

## Artificial Intelligence with Deep Learning



| Name          | GitHub profile |
| ---           | --- |
|Pawel Dymek    | [pdymek](https://github.com/pdymek)|
|Nikolai Pchelin| [Niko1707](https://github.com/Niko1707) |
|Nil Oller      | [NilOller](https://github.com/NilOller)|
|Francesc Fons  | [ffons9](https://github.com/ffons9)|


---
## Menu

- [Project description](#project-description)
   - [Hypothesis](#hypothesis)
   - [Dataset description](#dataset-description)
   - [System architecture](#system-architecture)
   - [Neural network architecture](#neural-network-architecture)
- Results
   - [Results of experiments](#results-of-experiments)
   - [Tensorboard output](#tensorboard-output)
   - [Visulalization](#visualization)
   - [Conclusions](#conclusions)
- [Project run instructions](#project-run-instructions)
	- [Prepare KITTI dataset catalog](#prepare-kitti-dataset-catalog)
	- [Build the virutal environment](#build-the-virutal-environment)
	- [Train](#train)
	- [Test](#test)
	- [Visualization](#visualization)
- [References](#references)

---










## Project description


### Hypothesis

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
### Dataset description

The data should be located in a single dierectory with below structure:
```
  .../<directory_name>/sequences/
                                /<scence_number>/velodyne
                                                /labels
                                ...
```

In our example it will be like:
```
  .../<directory_name>/sequences/
                                /00/velodyne
                                   /labels
                                /01/velodyne
                                   /labels
                                ...
```

![point_cloud_format](/F1_Documentation/docs/imgs/point_cloud_format.jpg)


![point_cloud_visualization_graphic](/F1_Documentation/docs/imgs/visualization.JPG)
---
### System architecture

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

![system_architecture_graphic](/F1_Documentation/docs/imgs/system_architecture.JPG)

---

### Neural network architecture

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

![point_net_architecture_graphic](/F1_Documentation/docs/imgs/point_net_architecture.JPG)


![](/F1_Documentation/docs/gifs/pth%20generation.gif)

--- 
## Results
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

--- 

### Results of experiments

--- 

### Tensorboard output

--- 

### Visulalization

<details><summary><b>🔍 Ground truth visualization</b></summary>

Ground truth:  

![](/F1_Documentation/docs/gifs/Ground%20truth.gif)

</details>

<details><summary><b>🔍 Training results visualization</b></summary>

Training results:  

![](/F1_Documentation/docs/gifs/Training%20result.gif)

</details>
--- 

### Conclusions

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

--- 




## Project run instructions

---
### Prepare KITTI dataset catalog

---

### Build the virutal environment

To build the virtual environment from the terminal. For instance, you can do it manually from the powershell in Visual Studio Code by running the following instructions:

```
    python -m venv virtualenv
    virtualenv\Scripts\activate.bat
    pip install -r requirements.txt
```

---  

### Train
Run the program through the following command from the terminal, hparamDatasetPath is relative path from main.py folder !

> python main.py --hparamDatasetPath B0_Dataset --hparamDatasetSequence 00 --hparamActionType train

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

---
### Test

```
  .../<directory_name>/sequences/
                                /<scence_number>/velodyne
                                                /labels
                                                /predictions
                                ...
```

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

---

### Visualization

Execute the following commands from the project root in the terminal to visualize both point cloud and voxels:

> python main.py --hparamDatasetPath E:\Project\Lidar_KITI\kitti\dataset --hparamDatasetSequence 11 --hparamPredictionsPath E:\Project\Lidar_KITI\kitti\dataset --hparamActionType visualize


Used parameters
- `--hparamActionType` - `"visualize"`
- `--hparamDatasetPath` - path for root dataset catalog
- `--hparamDatasetSequence` - number of visualized sequence 
- `--hparamPredictionsPath` - path for a directory with predictions

---
## References

At different stages of our project, we are referencing some other repositories and websites. They are inspirations about theoretical approaches, problem-solving, and also for some code.

REFERENCES
- Dataset:
   - https://www.cvlibs.net/datasets/kitti/
   - http://www.semantic-kitti.org/
- Neural Network:
   - https://github.com/Yvanali/KITTISeg
   - https://github.com/fxia22/pointnet.pytorch
   - https://github.com/marionacaros/3D-object-segmentation-light-PointNet
- Point Cloud Visualization Tool:
   - https://github.com/PRBonn/semantic-kitti-api
- Reporting:
   - https://www.tensorflow.org/tensorboard?hl=es-419




