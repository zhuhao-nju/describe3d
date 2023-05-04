import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Fully_Connected_Layer(torch.nn.Module):
    def __init__(self,in_features,out_features,last_fc=False):
        super(Fully_Connected_Layer, self).__init__()

        self.last_fc = last_fc
        self.in_features = in_features
        self.out_features = out_features
        self.activation = nn.LeakyReLU() if not last_fc else None
        # self.activation = nn.ReLU()
        self.L = nn.Linear(self.in_features,self.out_features)
        # self.dropout = nn.Dropout(p=0.4) if not last_fc else None

    def forward(self,x):
        x = self.L(x)
        x = self.activation(x) if not self.last_fc else x
        # x = self.activation(x)

        return x

class MLP(torch.nn.Module):

    def __init__(self,num_layers = 8,input_size=512,noise_size=0,output_size=192):
        super(MLP,self).__init__()

        self.num_layers = num_layers
        size_list = [input_size+noise_size] + num_layers * [output_size]
        for idx in range(num_layers):
            in_size = size_list[idx]
            out_size = size_list[idx+1]
            layer = Fully_Connected_Layer(in_size,out_size,last_fc=idx==num_layers-1)
            setattr(self,f'fc{idx}',layer)

    def forward(self,x):

        # noise = torch.normal(mean=0.012379, std=0.434819, size=x.shape).cuda()
        # x = torch.cat((x,noise),dim=-1)
        for idx in range(self.num_layers):
            layer = getattr(self,f'fc{idx}')
            x = layer(x)

        return x
