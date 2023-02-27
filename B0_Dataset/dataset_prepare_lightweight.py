import os
import numpy as np
from torch.utils import data
import yaml 
# from functions import get_kitti

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
    