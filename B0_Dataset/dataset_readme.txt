#######################################################
# POINT CLOUD - Data Structure Information            #
#######################################################

1.The 'velodyne' folders in the dataset contain the point clouds for each scan in each sequence.

Each '.bin' scan file is a list of points of the full pointcloud in [x,y,z,r] format:

  {x,y,z} = 3D coordinates. Data types: (float32, float32, float32)
  {r} = The fourth parameter, named Remission or Reflectance, is a measurement of diffuse reflection on surfaces, expressed as a percentage. Data type: (float32)

Remission indicates the proportion of the light that is diffusely reflected based on the proportion that would be reflected in a reference value (defined white).
Thus, 100% remission does not mean that all of the light is reflected, but rather that exactly the same amount of light is reflected as in the defined white surface.



2.The 'labels' folders in the dataset contains the labels for each scan in each sequence. 

Each '.label' file is a list of label information for each point {x,y,z,r} of the '.bin' point cloud keeping the following format:

  uint32 label = uint16 inst_label & uint16 sem_label

That is, for each scan XXXXXX.bin of the velodyne folder in the sequence folder of the original KITTI Odometry Benchmark, it is provided a file XXXXXX.label in the labels folder that contains for each point a label in binary format.

  {sem_label} = pointcloud labeled (semantic label).  Data type: (uint16)
  {inst_label} = pointcloud colorized with the color of each semantic label.  Data type: (uint16)

The label is a 32-bit unsigned integer (uint32) for each point, where the lower 16 bits correspond to the label. The upper 16 bits encode the instance id, which is temporally consistent over the whole sequence, i.e., the same object in two different scans gets the same id. This also holds for moving cars, but also static objects seen after loop closures.

  sem_label  = label & 0xFFFF  # semantic label in lower half of uint32
  inst_label = label >> 16     # instance id in upper half of uint32



3.Example of folders structure with pointcloud files XXXXXX.bin and XXXXXX.label:

sequences/00/velodyne/000000.bin --> point 0x0000000: {float32 x, float32 y, float32 z, float32 r}
            |                        point 0x0000001: {float32 x, float32 y, float32 z, float32 r}
            |                        ...
            |                        point 0x001E6FB: {float32 x, float32 y, float32 z, float32 r}
            |
            /labels/000000.label --> label point 0x0000000: {uint32 label}
                                     label point 0x0000001: {uint32 label}
                                     ...
                                     label point 0x001E6FB: {uint32 label}


             
4.Example of code in Python to read the point cloud:

  import numpy as np

  pcBin = np.fromfile('./B0_Dataset/sequences/00/velodyne/000000.bin', '<f4')
  pcBin = np.reshape(pcBin, (-1,4))
  print(pcBin)

  pcLbl = np.fromfile('./B0_Dataset/sequences/00/labels/000000.label', 'uint32')
  pcLbl.reshape((-1))
  print(pcLbl)


