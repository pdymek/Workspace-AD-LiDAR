####################################################################################
# AUTONOMOUS DRIVING NN ENGINE BASED ON LiDAR POINT CLOUD DATA                     #
####################################################################################
# To digest and execute the commands.
####################################################################################
# import torch
import os
import sys 
from A0_Configuration import hyperparam
from B0_Dataset import dataset_prepare_lightweight
from C0_Training      import train
from C1_Inference     import test



binFilesPath = r'E:\Project\Lidar_KITI\kitti\dataset\sequences\00\velodyne' # bin files path
lblFilesPath = r'E:\Project\Lidar_KITI\kitti\dataset\sequences\00\labels' # labels files path
binLwFilesPath = r'E:\Project\Lidar_KITI\kitti\dataset\sequences\00\velodyne_lw' # bin files path
lblLwFilesPath = r'E:\Project\Lidar_KITI\kitti\dataset\sequences\00\labels_lw' # labels files path

def main(args):
    if args.hparamActionType == 'train':
        dataset_prepare_lightweight.CreateSequenceLightweightPointCloud(binFilesPath, lblFilesPath,binLwFilesPath,lblLwFilesPath)
        train.train(args)
    elif args.hparamActionType == 'test':
        test.test(args)
    elif args.hparamActionType == 'visualize':
        os.chdir("./F0_Visualization/semantic-kitti-api")
        #cmndline = 'visualize.py --dataset ' + args.hparamDatasetPath + ' --config config/semantic-kitti.yaml --sequence ' + args.hparamDatasetSequence
        #print(cmndline)
        #os.system(cmndline) # call to visualize.py
        if args.hparamPredictionsPath != None:
            cmndline = 'visualize.py --dataset ' + args.hparamDatasetPath + ' --config config/semantic-kitti.yaml --sequence ' + args.hparamDatasetSequence + ' --predictions ' + args.hparamPredictionsPath
            print(cmndline)
            os.system(cmndline) # call to visualize.py
        cmndline = 'visualize_voxels.py --dataset ' + args.hparamDatasetPath + ' --sequence ' + args.hparamDatasetSequence 
        print(cmndline)
        os.system(cmndline) # call to visualize_voxels.py
        os.chdir("../..")
        
        
if __name__ == '__main__':
    parser = hyperparam.Parsing()
    args = parser.parse_args()
    main(args)

#





