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

        self.conv_1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1)
        #self.conv_1 = nn.Conv1d(input_dim, 64, 1)
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
        # x = [32,4000,64]
        num_points = x.shape[1] # torch.Size([BATCH = hparam, SAMPLES= 4000, DIMS = 3]) 
        x = x.transpose(2, 1) # [batch = 32, dims = 64, n_points = 4000]
        x = F.relu(self.bn_1(self.conv_1(x))) # max(0, x)
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x))) # [32,1024,4000]

        x = nn.MaxPool1d(kernel_size=x.shape[2])(x) # FeatureTNET = [32,1024,1]
        x = x.view(-1, 1024) # Features
        #print(x.shape) # [32,1024]

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)
        #print(x.shape) # [32,4096]
      
        identity_matrix = torch.eye(int(self.output_dim))
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, int(self.output_dim), int(self.output_dim)) + identity_matrix
        #print(x.shape) # Input_TF[32,3,3] & Feature TF[32,64,64]
        return x


class BasePointNet(nn.Module):

    def __init__(self, mdl_hparamPointDim=3, return_local_features=False):
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
        #self.conv_5 = nn.Conv1d(128, 1088, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)
        #self.bn_5 = nn.BatchNorm1d(1088)

    def forward(self, x):
        # x = [32,3,4000]
        num_points = x.shape[2]  # torch.Size([BATCH, DIMS, Samples])

        # x_tnet = x[:, :, :3]  # only apply T-NET to x and y and z
        x_tnet = x[:, :3, :]  # only apply T-NET to x and y and z #fixme #[32,3,3]
        x_tnet = x_tnet.transpose(2,1) #[32,4000,3]
        input_transform = self.input_transform(x_tnet) #[32,3,3]
        x_tnet = torch.bmm(x_tnet, input_transform)  # Performs a batch matrix-matrix product [32,4000,3]
        extra_dim = x[:, 3, :].unsqueeze(-1)  # add a new dimension (r)
        x_tnet = torch.cat([x_tnet, extra_dim], dim=2)  # x and y concat with z and r (reflection) [32,3,4]
        x_tnet = x_tnet.transpose(2, 1)  # [batch = 32, dims = 4, n_points = 4000]
        

        x = F.relu(self.bn_1(self.conv_1(x_tnet))) # [32,64,4000]
        x = F.relu(self.bn_2(self.conv_2(x))) # [""]
        x = x.transpose(2, 1)  # [32, 4000, 64]

        feature_transform = self.feature_transform(x)  #Outs--> [32, 64, 64]

        x = torch.bmm(x, feature_transform) #[32,4000,64]
        local_point_features = x  #[32,4000,64]

        x = x.transpose(2, 1) # [32,64,4000]
        x = F.relu(self.bn_3(self.conv_3(x))) # [32,64,4000]
        x = F.relu(self.bn_4(self.conv_4(x))) # [32,128,4000]
        x = F.relu(self.bn_5(self.conv_5(x))) # [32,1024,4000]
        x = nn.MaxPool1d(kernel_size=x.shape[2])(x) # [32,1024,1]
        global_feature = x.view(-1, 1024)  # [32, 1024]
        #print(global_feature.shape)

        if self.return_local_features:
            global_feature = global_feature.view(-1, 1024, 1).repeat(1, 1, num_points) #[32,1024,4000]
            global_feature = global_feature.transpose(2, 1)  # [32, 4000, 1024]
            #local_point_features = local_point_features # [32, 4000, 64] 
            

            #print(f"concat: {torch.cat([local_point_features, global_feature], dim=2).shape}\n, Feature_tranzsfom: {feature_transform.shape}")
            return torch.cat([local_point_features, global_feature], dim=2), feature_transform
        else:
            return global_feature, feature_transform


class SegmentationPointNet(nn.Module):

    def __init__(self, mdl_hparamNum_classes, mdl_hparamPointDim=3):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(return_local_features=True, mdl_hparamPointDim=3)

        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, mdl_hparamNum_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x = [32,3,4000]
        local_global_features, feature_transform = self.base_pointnet(x) #local_global_features.shape = [32,1216,4000] & feature_transform.shape = [32,64,64]
        x = local_global_features #[32, 4000, 1088]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x))) 
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x))) # [32,128, 4000 ]

        x = self.conv_4(x) # [32,20,4000]
        x = x.transpose(2, 1)

        return F.log_softmax(x, dim=-1), feature_transform

def Modeling():
    #TODO
    print("Modeling executed!")