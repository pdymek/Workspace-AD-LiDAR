####################################################################################
# HLD BUILDING BLOCK: INFERENCE                                                    #
####################################################################################
# TODO
# Run the test.
# Compute the metrics (e.g. accuracy) obtained.
####################################################################################

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from B0_Dataset.dataset import SemanticKittiDataset
from D0_Modeling.model import SegmentationPointNet
from torch.utils.data import DataLoader
from A0_Configuration.hyperparam import opt
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import shutil
import yaml

test_dataset = SemanticKittiDataset(
    dst_hparamDatasetPath=opt.hparamDatasetPath[0],
    dst_hparamDatasetSequence=opt.hparamTestDatasetSequence,
    dst_hparamYamlConfigPath=opt.hparamYamlConfigPath[0],
    dst_hparamNumberOfRandomPoints=False,
    dst_hparamActionType='test') 

test_dataloader = DataLoader(
    dataset = test_dataset,
    batch_size=1,
    shuffle=False)

num_classes=opt.hparamNumberOfClasses
feature_transform=opt.hparamFeatureTransform

# Load model from .pth
model = SegmentationPointNet(num_classes, feature_transform)
model.load_state_dict(torch.load(opt.hparamModelPthPath, map_location=torch.device('cpu')))
model.eval()

# Preprare predictions env
predictions_path = os.path.join(opt.hparamDatasetPath[0], opt.hparamTestDatasetSequence, 'predictions')

if os.path.exists(predictions_path):
    # os.remove(predictions_path)
    shutil.rmtree(predictions_path)
os.mkdir(predictions_path)

with open(opt.hparamYamlConfigPath[0], 'r') as stream:
    yaml_config = yaml.safe_load(stream)

learning_map_inv = yaml_config['learning_map_inv']

# Testing loop
for i, data in enumerate(test_dataloader):
    
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.to(opt.hparamDeviceType), target.to(opt.hparamDeviceType)
    
    model = model.eval()
    
    pred, feat_trans = model(points)
    pred = pred.view(-1, num_classes)
    # target = target.view(-1, 1)[:, 0]
    # loss = F.nll_loss(pred, target)
    pred_choice = pred.data.max(1)[1]
    # correct = pred_choice.eq(target.data).cpu().sum()
    
    # print(f"target: {target}, pred: {pred}, pred_choice: {pred_choice}, correct: {correct}")

    #Save predictions
    pred_choice_tmp = [learning_map_inv[k] for k in pred_choice.numpy().tolist()]
    pred_choice_conv = np.array(pred_choice_tmp).astype(np.uint32)
    file_name = os.path.basename(test_dataloader.dataset.pc_files[i])
    pred_file_path = os.path.join(
        predictions_path,
        file_name.replace('.bin', '.label')
    )
    pred_choice_conv.tofile(pred_file_path)

    
    
# shape_ious = []
# for i,data in tqdm(enumerate(test_dataloader, 0)):
#     points, target = data
#     points = points.transpose(2, 1)
#     points, target = points.to(opt.hparamDeviceType), target.to(opt.hparamDeviceType)
#     model = model.eval()
#     pred, _,= model(points)
#     pred_choice = pred.data.max(2)[1]

#     pred_np = pred_choice.cpu().data.numpy()
#     target_np = target.cpu().data.numpy() - 1 

#     for shape_idx in range(target_np.shape[0]):
#         parts = range(num_classes)#np.unique(target_np[shape_idx])
#         part_ious = []
#         for part in parts:
#             I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             if U == 0:
#                 iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             part_ious.append(iou)
#         shape_ious.append(np.mean(part_ious))

# print("mIOU for class {}: {}".format(opt.hparamClassChoice, np.mean(shape_ious)))





# Nikolai17:12
# ./visualize.py --sequence 00 --dataset /path/to/kitti/dataset/
# Nikolai17:20
# python visualize.py --sequence 00 --dataset /Users/nikolai/Downloads/UPC/VSC/Project/dataset/
# python visualize_mos.py --sequence 00 --dataset /Users/nikolai/Downloads/UPC/VSC/Project/dataset/


# " ./visualize.py --sequence 00 --dataset /path/to/kitti/dataset/ --predictions /path/to/your/predictions
# " ./visualize.py --sequence 00 --dataset /path/to/kitti/dataset/ --predictions /path/to/your/predictions    
    
    