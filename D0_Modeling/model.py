####################################################################################
# HLD BUILDING BLOCK: MODELING                                                     #
####################################################################################
# TODO
# Define the NN model.
####################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


# This module is using hyperparameters:
# mdl_hparamPointDim the default value should be equal to 3. (We're already taking into account x,y,z and r putting 3 as the value.)
# mdl_hparamNum_classes the value of it depends on the num_classes we want to segment.


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1] # torch.Size([BATCH = hparam, SAMPLES= 64, DIMS = 3]) 
        x = x.transpose(2, 1) # [batch = 63, dims = 3, n_points = 64]
        x = F.relu(self.bn_1(self.conv_1(x))) # max(0, x)
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x) 
        x = x.view(-1, 1024) # Features

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)
        
      
        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):

    def __init__(self, mdl_hparamPointDim, return_local_features=False):
        super(BasePointNet, self).__init__()
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=mdl_hparamPointDim, output_dim=mdl_hparamPointDim)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)

        # self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_1 = nn.Conv1d(4, 64, 1)  # Changed from 3 ch to 4 channels due to take reflection of light into account
        # self.conv_1 = nn.Conv1d(3, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        num_points = x.shape[1]  # torch.Size([BATCH = 63, SAMPLES = 64, DIMS = 3])

        x_tnet = x[:, :, :3]  # only apply T-NET to x and y and z
        input_transform = self.input_transform(x_tnet)
        x_tnet = torch.bmm(x_tnet, input_transform)  # Performs a batch matrix-matrix product
        x_tnet = torch.cat([x_tnet, x[:, :, 3].unsqueeze(2)], dim=2)  # x and y concat with z and r (reflection)
        x_tnet = x_tnet.transpose(2, 1)  # [batch, dims = 4, n_points]

        x = F.relu(self.bn_1(self.conv_1(x_tnet)))
        x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 4, 64]
        x = x.transpose(2, 1)  # [batch, 64, 4]

        feature_transform = self.feature_transform(x)  # [batch, 64, 64]

        x = torch.bmm(x, feature_transform)
        local_point_features = x  # [batch, 64, 64]

        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = nn.MaxPool1d(num_points)(x)
        global_feature = x.view(-1, 1024)  # [ batch, 1024, 1]

        if self.return_local_features:
            global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([global_feature.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return global_feature, feature_transform


class SegmentationPointNet(nn.Module):

    def __init__(self, mdl_hparamNum_classes, mdl_hparamPointDim):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(return_local_features=True, point_dimension=mdl_hparamPointDim)

        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, mdl_hparamNum_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

    def forward(self, x):
        local_global_features, feature_transform = self.base_pointnet(x)
        x = local_global_features
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = self.conv_4(x)
        x = x.transpose(2, 1)

        return F.log_softmax(x, dim=-1), feature_transform

def Modeling():
    #TODO
    print("Modeling executed!")
