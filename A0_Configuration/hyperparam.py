####################################################################################
# HLD BUILDING BLOCK: CONFIGURATION                                                #
####################################################################################
# To parse the command line arguments into the parameters of the system.
# Assignment to variables (torch.device, seed, etc.) of all the building blocks.
####################################################################################
import argparse
    
def Parsing():

    parser = argparse.ArgumentParser(description='AD/ADAS Lidar-based NN')
    
    # ENGINE (main.py) building block:
    parser.add_argument('--hparamDeviceType',            type=str,   required=False, default='cpu',               help = 'device type: cpu | gpu')
    parser.add_argument('--hparamSeedValue',             type=int,   required=False, default=123,                 help = 'seed value for reproducibility of experiments (default: 123)') 
    parser.add_argument('--hparamActionType',            type=str,   required=True, default='visualize',         help = 'action to execute: train | test | visualize | train&test | train&visualize | test&visualize | train&test&visualize (default: visualize)')

    # DATASET building block:
    parser.add_argument('--hparamDatasetPath',           type=str,   required=True,                               help = 'dataset path')
    parser.add_argument('--hparamDatasetSequence',       type=str,   required=False, default='00',                help = 'dataset sequence: 00 | 01 | ... | 21 (default: 00)')
    parser.add_argument('--hparamYamlConfigPath',        type=str,   required=False, default='F0_Visualization/semantic-kitti-api/config/semantic-kitti.yaml',                              help = 'yaml config path') 
   

    # TRAINING building block:
    parser.add_argument('--hparamTrainBatchSize',        type=int,   required=False, default=16,                  help = 'input batch size for training (default: 64)')
    parser.add_argument('--hparamTrainNumEpochs',        type=int,   required=False, default=12,                  help = 'number of epochs to run in training (default: 12)')
    parser.add_argument('--hparamNumberOfClasses',       type=int,   required=False, default=20,                  help = 'number of predicting classes') 
    parser.add_argument('--hparamFeatureTransform',      action='store_true',        default=False,                help="use feature transform")
    parser.add_argument('--hparamNumberOfEpochs',        type=int,   required=False, default=250,                help = 'number of epochs') #INFO: Added
    parser.add_argument('--hparamValBatchSize',          type=int,   required=False, default=16,                  help = 'input batch size for validation (default: 64)')
    parser.add_argument('--hparamValDatasetSequence',    type=str,   required=False, default='00',                help = 'dataset sequence: 00 | 01 | ... | 21 (default: 00)')
    parser.add_argument('--hparamValNumEpochs',          type=int,   required=False, default=12,                  help = 'number of epochs to run in validation (default: 12)')
    parser.add_argument('--hparamOptimizerLearningRate', type=float, required=False, default=0.001,               help = 'learning rate (default: 0.001)')
 
    # INFERENCE building block:
    parser.add_argument('--hparamTestDatasetSequence',   type=str,   required=False, default='00',            help = 'test dataset sequence: 00 | 01 | ... | 21 (default: 00)')
    parser.add_argument('--hparamTestBatchSize',         type=int,   required=False, default=1000,                help = 'input batch size for testing (default: 1000)')
    parser.add_argument('--hparamModelPthPath',          type=str,   required=False,                               help = 'model pth path')
    
    # MODELING building block:
    parser.add_argument('--hparamNumPoints',             type = int,  required = False, default= 4000,            help = 'PointNet number of points (n)')
    parser.add_argument('--hparamNumClasses',            type = int,  required = False, default= 3,                help = 'PointNet number of classes (k)')
    parser.add_argument('--hparamNumSemCategories',      type = int,  required = False, default= 64,              help = 'PointNet number of semantic categories (m)')
    parser.add_argument('--hparamPointDimension',        type = int,  required = False, default= 3,              help = 'Point Dimension used for T-NET therefore used in BasePointNet and Segmentation ')
    
    # VISUALIZATION building block:
    parser.add_argument('--hparamPredictionsPath',       type=str,   required=False, default=None,                 help = 'path to the predictions (.label files)')

    return parser
