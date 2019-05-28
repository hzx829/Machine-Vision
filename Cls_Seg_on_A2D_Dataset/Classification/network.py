import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math

class net(nn.Module):
    ####
    # define your model
    ####
    def __init__(self):
        super(net,self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features,43)
        self.bn = nn.BatchNorm1d(43,momentum=0.01)

    def forward(self,images):
        with torch.no_grad():
            features = self.resnet(images)

        features = features.reshape(features.size(0),-1)
        features = self.bn(self.linear(features))
        #features = self.linear(features)
        return features
