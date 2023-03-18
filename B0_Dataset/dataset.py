# ####################################################################################
# # HLD BUILDING BLOCK: DATASET                                                      #
# ####################################################################################
# # Handling of the selected dataset.
# ####################################################################################
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
                 dst_hparamYamlConfigPath: str,
                 dst_hparamPointDimension: int = 4) -> None:
        
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
        self.dst_hparamPointDimension = dst_hparamPointDimension
        
        
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
        if self.dst_hparamNumberOfRandomPoints:            
            sampling_indices = np.random.choice(pc_data.shape[0], self.dst_hparamNumberOfRandomPoints)            
            pc_data = pc_data[sampling_indices, :]
        
        labels = labels.astype(np.int64)
        #labels = labels.astype(np.uint8)
        if self.dst_hparamNumberOfRandomPoints:
            labels = labels[sampling_indices, :]
        
        output = (pc_data[:, :self.dst_hparamPointDimension], labels.reshape(-1))
        
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


