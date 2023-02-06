####################################################################################
# AUTONOMOUS DRIVING NN ENGINE BASED ON LiDAR POINT CLOUD DATA                     #
####################################################################################
# TODO
# To digest and execute the commands.
####################################################################################
import torch
import os
 
from A0_Configuration import hyperparam
from B0_Dataset       import dataset
from B1_Dataloader    import dataloader
from C0_Training      import train
from C1_Validation    import evaluate
from C2_Inference     import test
from D0_Modeling      import model
from E0_Detection     import detect
from E1_Segmentation  import segment
from F0_Visualization import graphics
from F1_Documentation import report


def main():

    args=hyperparam.Parsing()

    if args.hparamDeviceType == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset.AccessData()
    
    dataloader.Dataloader()
    
    train.Training()
    
    evaluate.Evaluation()
    
    test.Testing()
    
    model.Modeling()
    
    detect.Detection()
    
    segment.Segmentation()
    
    if args.hparamAction == 'visualize':
        # Debug_AD_LiDAR: Check current working directory.
        # retval = os.getcwd() #Debug_AD_LiDAR
        # print("Current working directory %s" %retval) #Debug_AD_LiDAR
        os.chdir("./F0_Visualization/semantic-kitti-api")
        os.system('visualize.py --dataset ../../B0_Dataset --config config/semantic-kitti.yaml --sequence %s' %args.hparamDatasetSequence)
        os.system('visualize_voxels.py --dataset ../../B0_Dataset --sequence %s' %args.hparamDatasetSequence)
        graphics.Visualization()
        os.chdir("../..")
    
    report.Reporting()



if __name__ == '__main__':
    main()









