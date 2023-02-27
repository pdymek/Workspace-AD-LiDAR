class opt():
    # hparamDatasetPath = r"/Users/nikolai/Downloads/UPC/VSC/Project/dataset/sequences",
    # hparamYamlConfigPath = "/Users/nikolai/Downloads/UPC/VSC/Project/Workspace-AD-LiDAR/F0_Visualization/semantic-kitti-api/config/semantic-kitti.yaml",
    hparamDatasetPath = r"G:\01_DATA\022_UPC\Project\_kitti_test\data_odometry_velodyne\dataset\sequences",
    hparamYamlConfigPath = "F0_Visualization\semantic-kitti-api\config\semantic-kitti.yaml",
    hparamNumberOfRandomPoints = 4000
    hparamNumberOfClasses = 20
    hparamClassChoice = 'bus'
    hparamDatasetSequence = '04'
    hparamBatchSize = 32
    hparamNumberOfEpochs = 100 #TODO: add to config ?
    hparamOutputFolder = 'output' #TODO: add to config ?
    hparamDeviceType = 'cpu'
    hparamFeatureTransform = False
    
    
    
    # from __future__ import print_function
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
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

blue = lambda x: '\033[94m' + x + '\033[0m'

torch.manual_seed(123)

train_dataset = SemanticKittiDataset(
    data_catalog_path=opt.hparamDatasetPath[0],
    sequence_number=opt.hparamDatasetSequence,
    yaml_config_path=opt.hparamYamlConfigPath[0],
    n_points=opt.hparamNumberOfRandomPoints,
    action_type='train')

val_dataset = SemanticKittiDataset(
    data_catalog_path=opt.hparamDatasetPath[0],
    sequence_number=opt.hparamDatasetSequence,
    yaml_config_path=opt.hparamYamlConfigPath[0],
    n_points=opt.hparamNumberOfRandomPoints,
    action_type='val')

test_dataset = SemanticKittiDataset(
    data_catalog_path=opt.hparamDatasetPath[0],
    sequence_number=opt.hparamDatasetSequence,
    yaml_config_path=opt.hparamYamlConfigPath[0],
    n_points=opt.hparamNumberOfRandomPoints,
    action_type='test')

train_dataloader = DataLoader_(
    dataset = train_dataset,
    batch_size=opt.hparamBatchSize,
    shuffle=True)

val_dataloader = DataLoader_(
    dataset = val_dataset,
    batch_size=opt.hparamBatchSize,
    shuffle=True)

test_dataloader = DataLoader_(
    dataset = test_dataset,
    batch_size=opt.hparamBatchSize,
    shuffle=True)

print(len(train_dataset), len(val_dataset), len(test_dataset))
print('classes', opt.hparamNumberOfClasses)

#try:
#    os.makedirs(opt.hparamOutputFolder)
#except OSError:
#    pass

num_classes=opt.hparamNumberOfClasses
feature_transform=opt.hparamFeatureTransform
model = SegmentationPointNet(num_classes, feature_transform)

#if opt.model != '':
#    model.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model = model.to(opt.hparamDeviceType)
#model.cuda()

num_batch = len(train_dataset) / opt.hparamBatchSize

for epoch in range(opt.hparamNumberOfEpochs):
    scheduler.step()
    for i, data in enumerate(train_dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.to(opt.hparamDeviceType), target.to(opt.hparamDeviceType)
        optimizer.zero_grad()
        model = model.train()
        pred, trans, trans_feat = model(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if opt.hparamFeatureTransform:
            loss += feature_transform(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.hparamBatchSize * 2500)))

        if i % 10 == 0:
            j, data = next(enumerate(test_dataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.to(opt.hparamDeviceType), target.to(opt.hparamDeviceType)
            model = model.eval()
            pred, _, _ = model(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.hparamBatchSize * 2500)))

    torch.save(model.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.hparamOutputFolder, opt.hparamClassChoice, epoch))

## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(test_dataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.to(opt.hparamDeviceType), target.to(opt.hparamDeviceType)
    model = model.eval()
    pred, _, _ = model(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.hparamClassChoice, np.mean(shape_ious)))