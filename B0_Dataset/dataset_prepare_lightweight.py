####################################################################################
# HLD BUILDING BLOCK: DATASET                                                      #
####################################################################################
# Dataset buider: toolset to adapt the KITTI-SemanticKITTI dataset to our use case.
####################################################################################
import numpy as np
import os

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
#  32, #: "motorcyclist"
#  40, #: "road"
#  44: "parking"
#  48, #: "sidewalk"
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
#  252, #: "moving-car"
#  253, #: "moving-bicyclist"
#  254, #: "moving-person"
#  255, #: "moving-motorcyclist"
#  256: "moving-on-rails"
#  257, #: "moving-bus"
#  258, #: "moving-truck"
#  259, #: "moving-other-vehicle"
]


def CreateLightweigthPointCloud(labelFile, binFile, labelLightweightFile, binLightweightFile):

    print('\nCreateLightweigthPointCloud start... ')

    file_stats = os.stat(labelFile)
    print(f'Label file size in bytes is {file_stats.st_size}')
    numPcLabelFile = np.int64(file_stats.st_size / 4)
    print(f'Label file size in PC points is {numPcLabelFile}')

    file_stats = os.stat(binFile)
    numPcBinFile = np.int64(file_stats.st_size / 16)
    print(f'Bin file size in bytes is {file_stats.st_size}')
    print(f'Bin file size in PC points is {numPcBinFile}')
    print(f'Processing Debug: {labelFile}& {binFile}')
    assert (numPcLabelFile == numPcBinFile) # assert only if evaluated condition is equal to false

    if numPcLabelFile != 0:
        fLbl = open(labelFile, "rb")
        fBin = open(binFile, "rb")
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
#            print(f'PC points counter: {numPcLabelFile}')
        fLbl.close()
        fBin.close()
        fLwLbl.close()
        fLwBin.close()

    print('...CreateLightweigthPointCloud end. ')


def CreateSequenceLightweightPointCloud(binFilesPath, lblFilesPath, binLwFilesPath, lblLwFilesPath):

    counterBinFilesInFolder = 0 # counter of files found inside a folder
    listBinFilesInFolder = [] #list of files found inside a folder
    for path in os.listdir(binFilesPath): # iterate through the folder
        if os.path.isfile(os.path.join(binFilesPath, path)): # check if current path is a file
            listBinFilesInFolder.append(path)
            counterBinFilesInFolder += 1
    print('Number of ".bin" files in folder [' + binFilesPath + '] = ', counterBinFilesInFolder)
    #print('Files in folder: ', listBinFilesInFolder)

    counterLblFilesInFolder = 0 # counter of files found inside a folder
    listBinFilesInFolder = [] #list of files found inside a folder
    for path in os.listdir(lblFilesPath): # iterate through the folder
        if os.path.isfile(os.path.join(lblFilesPath, path)): # check if current path is a file
            listBinFilesInFolder.append(path)
            counterLblFilesInFolder += 1
    print('Number of ".label" files in folder [' + lblFilesPath + '] = ', counterLblFilesInFolder)
    #print('Files in folder: ', listBinFilesInFolder)

    assert (counterBinFilesInFolder == counterLblFilesInFolder) # assert only if evaluated condition is equal to false

    binFilePath = []
    lblFilePath = []
    lblLwFilePath = []
    binLwFilePath = []

    for file in os.listdir(binFilesPath): # iterate through the folder
            binFilePath.append(os.path.join(binFilesPath, file)) # construct current name using path and file name
    print('Files in bin folder: ', binFilePath)

    for file in os.listdir(lblFilesPath): # iterate through the folder
            lblFilePath.append(os.path.join(lblFilesPath, file)) # construct current name using path and file name
    print('Files in labels folder: ', lblFilePath)

    for file in os.listdir(binFilesPath): # iterate through the folder
            #binLwFile = 'lw' + file
            binLwFile = file
            binLwFilePath.append(os.path.join(binLwFilesPath, binLwFile)) # construct current name using path and file name
    print('Files in lw bin folder: ', binLwFilePath)

    for file in os.listdir(lblFilesPath): # iterate through the folder
            #lblLwFile = 'lw' + file
            lblLwFile = file
            lblLwFilePath.append(os.path.join(lblLwFilesPath, lblLwFile)) # construct current name using path and file name
    print('Files in lw labels folder: ', lblLwFilePath)

    for counter in range(0, counterBinFilesInFolder-1):
        CreateLightweigthPointCloud(lblFilePath[counter], binFilePath[counter], lblLwFilePath[counter], binLwFilePath[counter])
