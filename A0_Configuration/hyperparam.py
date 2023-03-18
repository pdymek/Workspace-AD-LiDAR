####################################################################################
# HLD BUILDING BLOCK: CONFIGURATION                                                #
####################################################################################
# TODO
# To parse the command line arguments into the parameters of the system.
# Assignment to variables (torch.device, seed, etc.) of all the building blocks.
####################################################################################
import argparse
# class opt():
#     # hparamDatasetPath = r"/Users/nikolai/Downloads/UPC/VSC/Project/dataset/sequences",
#     # hparamYamlConfigPath = "/Users/nikolai/Downloads/UPC/VSC/Project/Workspace-AD-LiDAR/F0_Visualization/semantic-kitti-api/config/semantic-kitti.yaml",
#     hparamDatasetPath = r"E:\Project\Lidar_KITI\kitti\dataset\sequences",
#     hparamYamlConfigPath = "E:\Project\Workspace-AD-LiDAR-main_Pawel\Workspace-AD-LiDAR-main\F0_Visualization\semantic-kitti-api\config\semantic-kitti.yaml",
#     # hparamDatasetPath = r"G:\01_DATA\022_UPC\Project\_kitti_test\data_odometry_velodyne\dataset\sequences",
#     # hparamYamlConfigPath = "F0_Visualization\semantic-kitti-api\config\semantic-kitti.yaml",
#     hparamNumPoints = 4000
#     hparamNumberOfClasses = 20
#     hparamClassChoice = 'bus'
#     hparamDatasetSequence = '00'
#     hparamTrainBatchSize = 16
#     hparamValDatasetSequence = '08'
#     hparamValBatchSize = 8
#     hparamValNumberOfEpochs = 100
#     hparamTestDatasetSequence = '11'
#     hparamNumberOfEpochs = 100
#     hparamOutputFolder = 'E:\Project\Workspace-AD-LiDAR-main\Workspace-AD-LiDAR-main\Model_saved' 
#     hparamDeviceType = 'cpu'
#     hparamFeatureTransform = False
#     hparamModelPthPath = r"C:\Users\User\Downloads\seg_model_bus_7.pth"
#     hparamOptimizerLearningRate = 0.001
    
def Parsing():

    parser = argparse.ArgumentParser(description='AD/ADAS Lidar-based NN')
    # ENGINE (main.py) building block:
    parser.add_argument('--hparamDeviceType',            type=str,   required=False, default='cpu',               help = 'device type: cpu | gpu')
    parser.add_argument('--hparamSeedValue',             type=int,   required=False, default=123,                 help = 'seed value for reproducibility of experiments (default: 123)') #TODO unused?
    parser.add_argument('--hparamActionType',                type=str,   required=True, default='visualize',         help = 'action to execute: train | test | visualize | train&test | train&visualize | test&visualize | train&test&visualize (default: visualize)')
    # DATASET building block:
    # parser.add_argument('--hparamDatasetName',           type=str,   required=False, default='KITTI',             help = 'dataset name: KITTI | nuscenes | etc.')
    parser.add_argument('--hparamDatasetPath',           type=str,   required=True,                               help = 'dataset path')
    parser.add_argument('--hparamDatasetSequence',       type=str,   required=False, default='00',                help = 'dataset sequence: 00 | 01 | ... | 21 (default: 00)')
    # parser.add_argument('--hparamNumberOfRandomPoints',  type=int,   required=False, default=4000,                help = 'number of datapoints randomsampled in dataset class')
    
    # DATALOADER building block:
    parser.add_argument('--hparamOptimizerType',         type=str,   required=False, default='00',                help = 'optimizer type: Adam | SGD | RMSProp (default: Adam)') #TODO unused?
    parser.add_argument('--hparamOptimizerLearningRate', type=float, required=False, default=0.001,               help = 'learning rate (default: 0.001)') #TODO unused?
    # TRAINING building block:
    parser.add_argument('--hparamTrainBatchSize',        type=int,   required=False, default=16,                  help = 'input batch size for training (default: 64)')
    parser.add_argument('--hparamTrainNumEpochs',        type=int,   required=False, default=12,                  help = 'number of epochs to run in training (default: 12)')
    parser.add_argument('--hparamLossFunction',          type=str,   required=False, default='CrossEntropyLoss',  help = 'optimizer type: CrossEntropyLoss | L1Loss | MSELoss | NLLLoss | KLDivLoss (default: CrossEntropyLoss)')
    parser.add_argument('--hparamNumberOfClasses',       type=int,   required=False, default=20,                  help = 'number of predicting classes') #TODO: Should be as parameter or calculated from dataset?
    parser.add_argument('--hparamFeatureTransform',      action='store_true',        default=False,                help="use feature transform")
    parser.add_argument('--hparamClassChoice',           type=str,   required=False, default='bus') #class for train test run
   
    # VALIDATION building block:
    parser.add_argument('--hparamValBatchSize',        type=int,   required=False, default=16,                  help = 'input batch size for validation (default: 64)')
    parser.add_argument('--hparamValDatasetSequence',       type=str,   required=False, default='00',                help = 'dataset sequence: 00 | 01 | ... | 21 (default: 00)')
    parser.add_argument('--hparamValNumEpochs',        type=int,   required=False, default=12,                  help = 'number of epochs to run in validation (default: 12)')
 
 
    # INFERENCE building block:
    parser.add_argument('--hparamTestDatasetSequence',       type=str,   required=False, default='00',            help = 'test dataset sequence: 00 | 01 | ... | 21 (default: 00)')
    parser.add_argument('--hparamTestBatchSize',         type=int,   required=False, default=1000,                help = 'input batch size for testing (default: 1000)')
    parser.add_argument('--hparamModelPthPath',          type=str,   required=False,                               help = 'model pth path')
    
    
    # MODELING building block:
    parser.add_argument('--hparamModelType',             type=str,   required=False, default='pointnet',          help = 'NN model type: pointnet | pointnetlight') #TODO: unused?
    parser.add_argument('--hparamModelSave',             action='store_true',        default=False,               help = 'for saving the current trained model')
    parser.add_argument('--hparamModelPretrained',       action='store_true',        default=False,               help = 'to use pre-trained model')
    parser.add_argument('--hparamNumPoints',             type = int,  required = False, default= 4000,            help = 'PointNet number of points (n)')
    parser.add_argument('--hparamNumClasses',            type = int,  required = False, default= 3,                help = 'PointNet number of classes (k)')
    parser.add_argument('--hparamNumSemCategories',     type = int,  required = False, default= 64,              help = 'PointNet number of semantic categories (m)')
    parser.add_argument('--hparamPointDimension',     type = int,  required = False, default= 3,              help = 'Point Dimension used for T-NET therefore used in BasePointNet and Segmentation ')
    
    
    # DETECTION building block:
    parser.add_argument('--hparamNumberOfEpochs',         type=int,   required=False, default=250,                help = 'number of epochs') #INFO: Added
    parser.add_argument('--hparamNumberOfWorkers',         type=int,   required=False, default=4,                help = 'no of workers') #INFO: Added
    
    # SEGMENTATION building block:
 
    # VISUALIZATION building block:
    parser.add_argument('--hparamPredictionsPath',      type=str,   required=False, default=None,                 help = 'path to the predictions (.label files)')
    # DOCUMENTATION building block:
    
    parser.add_argument('--hparamYamlConfigPath',           type=str,   required=False, default='F0_Visualization/semantic-kitti-api/config/semantic-kitti.yaml',                              help = 'yaml config path') #TODO: It should be as paramter or we put that files into directory structure?

    # args = parser.parse_args()
    # print("Parsing executed!")
    # print(args)
    # return args
    return parser










