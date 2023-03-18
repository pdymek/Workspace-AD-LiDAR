


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

![point_cloud_format](/F1_Documentation/docs/imgs/point_cloud_format.jpg.JPG)

![system_architecture_graphic](/F1_Documentation/docs/imgs/system_architecture.JPG)

![point_net_architecture_graphic](/F1_Documentation/docs/imgs/point_net_architecture.JPG)


![point_cloud_visualization_graphic](/F1_Documentation/docs/imgs/visualization.JPG)


## Instructions
### Prepare KITTI dataset catalog

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


### Build the virutal environment

To build the virtual environment from the terminal. For instance, you can do it manually from the powershell in Visual Studio Code by running the following instructions:

```
    python -m venv virtualenv
    virtualenv\Scripts\activate.bat
    pip install -r requirements.txt
```

### Train
Run the program through the following command from the terminal, hparamDatasetPath is relative path from main.py folder !

> python main.py --hparamDatasetPath B0_Dataset --hparamDatasetSequence 00 --hparamActionType train


### Test

```
  .../<directory_name>/sequences/
                                /<scence_number>/velodyne
                                                /labels
                                                /predictions
                                ...
```
### Visualization

Execute the following commands from the project root in the terminal to visualize both point cloud and voxels:
NOTE: hparamDatasetPath is relative path from main.py folder !

> python main.py --hparamDatasetPath B0_Dataset --hparamDatasetSequence 00 --hparamActionType visualize


