####################################################################################
# HLD BUILDING BLOCK: CONFIGURATION                                                #
####################################################################################
# TODO
# To parse the command line arguments into the parameters of the system.
# Assignment to variables (torch.device, seed, etc.) of all the building blocks.
####################################################################################
import argparse

def Parsing():

    parser = argparse.ArgumentParser(description='AD/ADAS Lidar-based NN')
    # ENGINE (main.py) building block:
    parser.add_argument('--hparamDeviceType',            type=str,   required=False, default='cpu',               help = 'device type: cpu | gpu')
    parser.add_argument('--hparamSeedValue',             type=int,   required=False, default=123,                 help = 'seed value for reproducibility of experiments (default: 123)')
    parser.add_argument('--hparamAction',                type=str,   required=False, default='visualize',         help = 'action to execute: train | test | visualize | train&test | train&visualize | test&visualize | train&test&visualize (default: visualize)')
    # DATASET building block:
    parser.add_argument('--hparamDatasetName',           type=str,   required=False, default='KITTI',             help = 'dataset name: KITTI | nuscenes | etc.')
    parser.add_argument('--hparamDatasetPath',           type=str,   required=True,                               help = 'dataset path')
    parser.add_argument('--hparamDatasetSequence',       type=str,   required=False, default='00',                help = 'dataset sequence: 00 | 01 | ... | 21 (default: 00)')
    parser.add_argument('--hparamNumberOfRandomPoints',  type=int,   required=False, default=4000,                help = 'number of datapoints randomsampled in dataset class') #INFO: 
    # DATALOADER building block:
    parser.add_argument('--hparamOptimizerType',         type=str,   required=False, default='00',                help = 'optimizer type: Adam | SGD | RMSProp (default: Adam)')
    parser.add_argument('--hparamOptimizerLearningRate', type=float, required=False, default=0.001,               help = 'learning rate (default: 0.001)')
    # TRAINING building block:
    parser.add_argument('--hparamTrainBatchSize',        type=int,   required=False, default=16,                  help = 'input batch size for training (default: 64)')
    parser.add_argument('--hparamTrainNumEpochs',        type=int,   required=False, default=12,                  help = 'number of epochs to run in training (default: 12)')
    parser.add_argument('--hparamLossFunction',          type=str,   required=False, default='CrossEntropyLoss',  help = 'optimizer type: CrossEntropyLoss | L1Loss | MSELoss | NLLLoss | KLDivLoss (default: CrossEntropyLoss)')
    parser.add_argument('--hparamNumberOfClasses',       type=int,   required=False, default=34,                help = 'number of predicting classes') #TODO: Should be as parameter or calculated from dataset?
    # VALIDATION building block:
 
    # INFERENCE building block:
    parser.add_argument('--hparamTestBatchSize',         type=int,   required=False, default=1000,                help = 'input batch size for testing (default: 1000)')
    # MODELING building block:
    parser.add_argument('--hparamModelType',             type=str,   required=False, default='pointnet',          help = 'NN model type: pointnet | pointnetlight')
    parser.add_argument('--hparamModelSave',             action='store_true',        default=False,               help = 'for saving the current trained model')
    parser.add_argument('--hparamModelPretrained',       action='store_true',        default=False,               help = 'to use pre-trained model')
    parser.add_argument('--hparamNumPoints',             type = int,  required = False, default= 4000,            help = 'PointNet number of points (n)')
    parser.add_argument('--hparamNumClasses',            type = int,  required = False, default= 3,                help = 'PointNet number of classes (k)')
    parser.add_argument('--hparamNumSemCategories ',     type = int,  required = False, default= 64,              help = 'PointNet number of semantic categories (m)')
    
    # DETECTION building block:
    parser.add_argument('--hparamNumberOfEpochs',         type=int,   required=False, default=250,                help = 'number of epochs') #INFO: Added
    parser.add_argument('--hparamNumberOfWorkers',         type=int,   required=False, default=4,                help = 'no of workers') #INFO: Added
    
    # SEGMENTATION building block:
 
    # VISUALIZATION building block:
 
    # DOCUMENTATION building block:
    
    parser.add_argument('--hparamYamlConfigPath',           type=str,   required=True,                               help = 'yaml config path') #TODO: It should be as paramter or we put that files into directory structure?

    args = parser.parse_args()
    print("Parsing executed!")
    print(args)
    return args











