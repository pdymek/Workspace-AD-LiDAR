#######################################################
# PROJECT BUILDER INSTRUCTIONS FOR workspace_AD_LiDAR #
#######################################################

# STEP 0
# Update the KITTI/-SemanticKITTI dataset in B0_Dataset/sequences/ folder of your AD_LiDAR project workspace.
    #  root AD_LiDAR / B0_Dataset / sequences /
    #                                         / 00 / velodyne
    #                                              / labels
    #                                              / voxels
    #                                         / 01 / velodyne
    #                                              / labels
    #                                              / voxels
    #                                         / .. / velodyne
    #                                              / labels
    #                                              / voxels
    #                                         / 21 / velodyne
    #                                              / labels
    #                                              / voxels


# STEP 1
# To build the virtual environment from the terminal.
# For instance, you can do it manually from the powershell in Visual Studio Code by running the following instructions:

    python -m venv virtualenv
    virtualenv\Scripts\activate.bat
    pip install -r requirements.txt


# STEP 2
# Run the program through the following command from the terminal:
# NOTE: hparamDatasetPath is relative path from main.py folder !

    python main.py --hparamDatasetPath B0_Dataset --hparamDatasetSequence 00 --hparamActionType train


# STEP 3
# Visualization:
# Execute the following commands from the project root in the terminal to visualize both point cloud and voxels:
# NOTE: hparamDatasetPath is relative path from main.py folder !

    python main.py --hparamDatasetPath B0_Dataset --hparamDatasetSequence 00 --hparamActionType visualize





