import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from B0_Dataset.dataset import SemanticKittiDataset
from D0_Modeling.model import SegmentationPointNet
from B1_Dataloader.dataloader import DataLoader_
from A0_Configuration.hyperparam import opt
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

test_dataset = SemanticKittiDataset(
    dst_hparamDatasetPath=opt.hparamDatasetPath[0],
    dst_hparamDatasetSequence=opt.hparamDatasetSequence,
    dst_hparamYamlConfigPath=opt.hparamYamlConfigPath[0],
    dst_hparamNumberOfRandomPoints=10000,
    dst_hparamActionType='val') #TODO: keep it or not?

test_dataloader = DataLoader_(
    dataset = test_dataset,
    batch_size=1,
    shuffle=False)

num_classes=opt.hparamNumberOfClasses
feature_transform=opt.hparamFeatureTransform


model = SegmentationPointNet(num_classes, feature_transform)
model.load_state_dict(torch.load(opt.hparamModelPthPath, map_location=torch.device('cpu')))
model.eval()

for i, data in enumerate(test_dataloader):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.to(opt.hparamDeviceType), target.to(opt.hparamDeviceType)
    
    model = model.eval()
    
    pred, feat_trans = model(points)
    pred = pred.view(-1, num_classes)
    target = target.view(-1, 1)[:, 0]
    # loss = F.nll_loss(pred, target)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    
    print(f"target: {target}, pred: {pred}, pred_choice: {pred_choice}, correct: {correct}")


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