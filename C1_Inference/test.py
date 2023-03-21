####################################################################################
# HLD BUILDING BLOCK: INFERENCE                                                    #
####################################################################################
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
# from A0_Configuration.hyperparam import opt
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import shutil
import yaml

def test(opt):
    test_dataset = SemanticKittiDataset(
        dst_hparamDatasetPath=opt.hparamDatasetPath,
        dst_hparamDatasetSequence=opt.hparamTestDatasetSequence,
        dst_hparamYamlConfigPath=opt.hparamYamlConfigPath,
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
    predictions_path = os.path.join(opt.hparamDatasetPath, opt.hparamTestDatasetSequence, 'predictions')

    if os.path.exists(predictions_path):
        # os.remove(predictions_path)
        shutil.rmtree(predictions_path)
    os.mkdir(predictions_path)

    with open(opt.hparamYamlConfigPath, 'r') as stream:
        yaml_config = yaml.safe_load(stream)

    learning_map_inv = yaml_config['learning_map_inv']

    # Testing loop
    print("Start genearting predictions .labels")
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

    print('Generation predictions completed')        
    