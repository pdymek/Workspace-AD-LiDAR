####################################################################################
# AUTONOMOUS DRIVING NN ENGINE BASED ON LiDAR POINT CLOUD DATA                     #
####################################################################################
# To digest and execute the commands.
####################################################################################
import torch
import os
 
from A0_Configuration import hyperparam
#from B0_Dataset       import dataset
#from C0_Training      import train
#from C2_Inference     import test
from D0_Modeling      import model
from E1_Segmentation  import segment
from F0_Visualization import graphics
from F1_Documentation import report


def main():

    args=hyperparam.Parsing()

    if args.hparamDeviceType == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
   
    #dataloader.Dataloader()
    
    #train.Training()
    
    #test.Testing()
    
    model.Modeling()
      
    segment.Segmentation()
    
    if args.hparamActionType == 'visualize':
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
        graphics.Visualization()
        os.chdir("../..")
    
    report.Reporting()



if __name__ == '__main__':
    main()







