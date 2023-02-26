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

labelsLightweight = [ 
#  0 : "unlabeled"
#  1 : "outlier"
  10, #: "car"
  11, #: "bicycle"
  13, #: "bus"
  15, #: "motorcycle"
  16, #: "on-rails"
  18, #: "truck"
  20, #: "other-vehicle"
  30, #: "person"
  31, #: "bicyclist"
  32, #: "motorcyclist"
  40, #: "road"
#  44: "parking"
  48, #: "sidewalk"
#  49: "other-ground"
#  50: "building"
#  51: "fence"
#  52: "other-structure"
#  60: "lane-marking"
#  70: "vegetation"
#  71: "trunk"
#  72: "terrain"
#  80: "pole"
  81, #: "traffic-sign"
#  99: "other-object"
  252, #: "moving-car"
  253, #: "moving-bicyclist"
  254, #: "moving-person"
  255, #: "moving-motorcyclist"
#  256: "moving-on-rails"
  257, #: "moving-bus"
  258, #: "moving-truck"
  259, #: "moving-other-vehicle"
]


def CreateLightweigthPointCloud():
    labelFile = "./B0_Dataset/sequences/00/labels/000000.label"
    file_stats = os.stat(labelFile)
    print(f'Label file size in bytes is {file_stats.st_size}')
    numPcLabelFile = np.int64(file_stats.st_size / 4)
    print(f'Label file size in PC points is {numPcLabelFile}')
    binFile = "./B0_Dataset/sequences/00/velodyne/000000.bin"
    file_stats = os.stat(binFile)
    numPcBinFile = np.int64(file_stats.st_size / 16)
    print(f'Bin file size in bytes is {file_stats.st_size}')
    print(f'Bin file size in PC points is {numPcBinFile}')
    assert (numPcLabelFile == numPcBinFile) # assert only if evaluated condition is equal to false
    if numPcLabelFile != 0:
        fLbl = open(labelFile, "rb")
        fBin = open(binFile, "rb")
        labelLightweightFile = "./B0_Dataset/sequences/00/labels/lw0000.label"
        binLightweightFile = "./B0_Dataset/sequences/00/velodyne/lw0000.bin"
        fLwLbl = open(labelLightweightFile, "w+")
        fLwBin = open(binLightweightFile, "w+")
        while numPcLabelFile != 0:
            pcLbl = np.fromfile(fLbl, dtype=np.uint32, count=1)
#            print(f'Label: {pcLbl}'); 
            pcX, pcY, pcZ, pcR = np.fromfile(fBin,dtype='<f4', count=4) #little-endian float32 
            pointcloud = np.array([pcX,pcY,pcZ,pcR], dtype=np.float32)
#            print(f'Pointcloud: {pcX,pcY,pcZ,pcR}'); 
            if pcLbl in labelsLightweight:
                pcLbl.tofile(fLwLbl)
                pointcloud.tofile(fLwBin)
            numPcLabelFile = numPcLabelFile - 1
            print(f'PC points counter: {numPcLabelFile}')
        fLbl.close()
        fBin.close()
        fLwLbl.close()
        fLwBin.close()
    

class SemanticKittiDataset(data.Dataset):
    def __init__(self,
                 data_catalog_path: str,
                 sequence_number: str,
                 action_type: str,
                 n_points: int,
                 seg_classes: int,
                 yaml_config_path: str) -> None:
        
        """Kitti dataset construcotr

        Args:
            data_catalog_path (str): path for directory containing sequences
            sequence_number (str): 2 digit number of sequence
            action_type (str): train/test/val option
            n_points (int): required number of points
            yaml_config_path (str): path for yaml dataset opations
        """
        self.action_type = action_type
        self.scene = sequence_number
        self.data_catalog_path = data_catalog_path
        self.seg_classes = seg_classes
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
            labels = labels & 0xFFFF
            labels = np.vectorize(self.learning_map.__getitem__)(labels) 
        elif self.action_type in ['test']:
            labels = np.expand_dims(np.zeros_like(pc_data[:,0], dtype=int),
                                    axis=1)  
        sampling_indices = np.random.choice(pc_data.shape[0], self.n_points)            
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
        data_catalog_path=PATH, 
        sequence_number='04', 
        action_type='train',
        yaml_config_path=YAML_PATH, 
        n_points = 4000
    )
    
    y = next(iter(kd))   
    print('--------------')
    print('Single file data: shape: ', y[0].shape)     
    print('Single file data: sample: ', y[0][0])     
    print('Single file labels: shape: ', y[1].shape) 
    print('Single file labels: sample: ', y[1][0])         
    print('--------------')