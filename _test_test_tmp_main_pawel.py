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
    dst_hparamNumberOfRandomPoints=opt.hparamNumPoints,
    dst_hparamActionType='test')

test_dataloader = DataLoader_(
    dataset = test_dataset,
    batch_size=1,
    shuffle=False)

num_classes=opt.hparamNumberOfClasses
feature_transform=opt.hparamFeatureTransform


model = SegmentationPointNet(num_classes, feature_transform)
model.load_state_dict(torch.load(opt.hparamModelPthPath))
model.eval()

for i, data in enumerate(test_dataloader):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.to(opt.hparamDeviceType), target.to(opt.hparamDeviceType)
    
    model = model.eval()
    
    pred, feat_trans = model(points)
    pred = pred.view(-1, num_classes)
    target = target.view(-1, 1)[:, 0]
    loss = F.nll_loss(pred, target)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    
    print(f"target: {target}, pred: {pred}, pred_choice: {pred_choice}, correct: {correct}")