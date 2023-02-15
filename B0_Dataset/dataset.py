# ####################################################################################
# # HLD BUILDING BLOCK: DATASET                                                      #
# ####################################################################################
# # TODO
# # Handling of the selected dataset.
# ####################################################################################

# def AccessData():
#     #TODO
#     print("Dataset access executed!")


import os
import numpy as np
from torch.utils import data
import yaml 
# from functions import get_kitti_yaml_config # TODO Uncomment, if we have to move function to another module 

class SemanticKittiDataset(data.Dataset):
    def __init__(self,
                 data_catalog_path: str,
                 sequence_number: str,
                 action_type: str,
                 n_points: int = 4000,
                 yaml_config_path: str) -> None:
        
        """Kitti dataset construcotr

        Args:
            data_catalog_path (str): path for directory containing sequences
            sequence_number (str): 2 digit number of sequence
            action_type (str): train/test/val option
            n_points (int): ...
            yaml_config_path (str): path for yaml dataset opations
        """
        self.action_type = action_type
        self.scene = sequence_number
        self.data_catalog_path = data_catalog_path
        self.n_points = n_points
        self.yaml_config_path = yaml_config_path
        
        
        yaml_config = self._get_kitti_yaml_config(self.yaml_config_path)
        self.learning_map = yaml_config['learning_map']
        
        sequence_number_path = os.path.join(data_catalog_path,
                                   str(sequence_number).zfill(2),
                                   'velodyne')
        pc_files = []
        for dir_path, _, files in os.walk(sequence_number_path):
            for file in files:
                file_path = os.path.abspath(os.path.join(dir_path, file))
                pc_files.append(file_path)
        self.pc_files = pc_files
        
    def __len__(self):
        return len(self.pc_files)
    
    def __getitem__(self, index):
        pc_data = np.fromfile(self.pc_files[index], dtype=np.float32)
        pc_data = pc_data.reshape((-1, 4))
        if self.action_type in ['train', 'val']:
            labels = np.fromfile(
                self.pc_files[index] \
                    .replace('velodyne','labels') \
                    .replace('.bin', '.label'),
                dtype=np.int32
            )
            labels = labels.reshape((-1,1))
            labels = labels & 0xFFFF # TODO Check for other implementation method?
            labels = np.vectorize(self.learning_map.__getitem__)(labels) 
        elif self.action_type in ['test']:
            labels = np.expand_dims(np.zeros_like(pc_data[:,0], dtype=int),
                                    axis=1)            
        labels = labels.astype(np.uint8)
        output = (pc_data[:, :3], labels)
        
        
        sampling_indices = np.random.choice(output.shape[0], self.n_points)
        output = output[sampling_indices, :]
        
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
    PATH = r"G:\01_DATA\022_UPC\Project\_kitti_test\data_odometry_velodyne\dataset\sequences"
    YAML_PATH = "F0_Visualization\semantic-kitti-api\config\semantic-kitti.yaml"
    kd = SemanticKittiDataset(
        data_catalog_path=PATH, sequence_number=4, action_type='train', yaml_config_path=YAML_PATH
    )
    y = next(iter(kd))   
    print('--------------')
    print('Single file data: shape: ', y[0].shape)     
    print('Single file data: sample: ', y[0][0])     
    print('Single file labels: shape: ', y[1].shape) 
    print('Single file labels: sample: ', y[1][0])         
    print('--------------')
    
