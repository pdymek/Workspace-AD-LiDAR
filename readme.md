


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

### Visualization

Execute the following commands from the project root in the terminal to visualize both point cloud and voxels:
NOTE: hparamDatasetPath is relative path from main.py folder !

> python main.py --hparamDatasetPath B0_Dataset --hparamDatasetSequence 00 --hparamActionType visualize
