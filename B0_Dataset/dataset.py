# ####################################################################################
# # HLD BUILDING BLOCK: DATASET                                                      #
# ####################################################################################
# # Handling of the selected dataset.
# ####################################################################################

# def AccessData():
#     #TODO
#     print("Dataset access executed!")


import os
import numpy as np
from torch.utils import data
import yaml 

class SemanticKittiDataset(data.Dataset):
    def __init__(self,
                 dst_hparamDatasetPath: str,
                 dst_hparamDatasetSequence: str,
                 dst_hparamActionType: str,
                 dst_hparamNumberOfRandomPoints: int,
                 dst_hparamYamlConfigPath: str) -> None:
        
        """Kitti dataset construcotr

        Args:
            dst_hparamDatasetPath (str): path for directory containing sequences
            dst_hparamDatasetSequence (str): 2 digit number of sequence
            dst_hparamActionType (str): train/test/val option
            dst_hparamNumberOfRandomPoints (int): required number of points
            dst_hparamYamlConfigPath (str): path for yaml dataset opations
        """
        self.dst_hparamActionType = dst_hparamActionType
        self.scene = dst_hparamDatasetSequence
        self.dst_hparamDatasetPath = dst_hparamDatasetPath
        self.dst_hparamNumberOfRandomPoints = dst_hparamNumberOfRandomPoints
        self.dst_hparamYamlConfigPath = dst_hparamYamlConfigPath
        
        
        yaml_config = self._get_kitti_yaml_config(self.dst_hparamYamlConfigPath)
        self.learning_map = yaml_config['learning_map']
        
        dst_hparamDatasetSequence_path = os.path.join(dst_hparamDatasetPath,
                                   str(dst_hparamDatasetSequence).zfill(2),
                                   'velodyne')
        pc_files = []
        for dir_path, _, files in os.walk(dst_hparamDatasetSequence_path):
            for file in files:
                file_path = os.path.abspath(os.path.join(dir_path, file))
                pc_files.append(file_path)
        self.pc_files = pc_files
        
    def __len__(self):
        return len(self.pc_files)
    
    def __getitem__(self, index):
        pc_data = np.fromfile(self.pc_files[index], dtype=np.float32)
        pc_data = pc_data.reshape((-1, 4))
        if self.dst_hparamActionType in ['train', 'val']:
            labels = np.fromfile(
                self.pc_files[index] \
                    .replace('velodyne','labels') \
                    .replace('.bin', '.label'),
                dtype=np.int32
            )
            labels = labels.reshape((-1,1))
            labels = labels & 0xFFFF
            labels = np.vectorize(self.learning_map.__getitem__)(labels) 
        elif self.dst_hparamActionType in ['test']:
            labels = np.expand_dims(np.zeros_like(pc_data[:,0], dtype=int),
                                    axis=1)  
        sampling_indices = np.random.choice(pc_data.shape[0], self.dst_hparamNumberOfRandomPoints)            
        pc_data = pc_data[sampling_indices, :]
        
        labels = labels.astype(np.uint8)
        labels = labels[sampling_indices, :]
        
        output = (pc_data[:, :3], labels.reshape(-1))
        
        return output
    
    def _get_kitti_yaml_config(self, yaml_path: str) -> dict:
        """Read min kitti main configuration
            yaml_path(str)
        Returns:
            dict: dataset configuration
        """
        with open(yaml_path, 'r') as stream:
            yaml_config = yaml.safe_load(stream)
        return yaml_config    


####################################################################################################
# Remove in final project, just for data investingation purposes
if __name__ == "__main__":
    #PATH = r"G:\01_DATA\022_UPC\Project\_kitti_test\data_odometry_velodyne\dataset\sequences"
    PATH = r"/Users/nikolai/Downloads/UPC/VSC/Project/dataset/sequences"
    YAML_PATH = "F0_Visualization\semantic-kitti-api\config\semantic-kitti.yaml"
    
    kd = SemanticKittiDataset(
        dst_hparamDatasetPath=PATH, 
        dst_hparamDatasetSequence='04', 
        dst_hparamActionType='train',
        dst_hparamYamlConfigPath=YAML_PATH, 
        dst_hparamNumberOfRandomPoints = 4000
    )
    
    y = next(iter(kd))   
    print('--------------')
    print('Single file data: shape: ', y[0].shape)     
    print('Single file data: sample: ', y[0][0])     
    print('Single file labels: shape: ', y[1].shape) 
    print('Single file labels: sample: ', y[1][0])         
    print('--------------')