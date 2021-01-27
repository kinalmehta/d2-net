
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

import e2cnn
import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces   


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        N = 8
        self.gspace = gspaces.Rot2dOnR2(N)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * 16)
        self.layer = enn.R2Conv(self.in_type, self.out_type, 3,
                      stride=1,
                      padding=1,
                      dilation=1,
                      bias=True,
                      )
        self.invariant = enn.GroupPooling(self.out_type)
    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        out = self.layer(x)
        out = self.invariant(out)
        out = out.tensor
        return out

class ModelDilated(nn.Module):
    def __init__(self):
        super(ModelDilated, self).__init__()
        N = 8
        self.gspace = gspaces.Rot2dOnR2(N)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * 16)
        self.layer = enn.R2Conv(self.in_type, self.out_type, 3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      bias=True,
                      )
        self.invariant = enn.GroupPooling(self.out_type)
    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        out = self.layer(x)
        out = self.invariant(out)
        out = out.tensor
        return out


if __name__=="__main__":
    m = Model()
    md = ModelDilated()
    ip = torch.randn(1,3,100,100)
    op1 = m(ip)
    op2 = md(ip)

    totalParams = sum(p.numel() for p in m.parameters())
    totalParams2 = sum(p.numel() for p in md.parameters())
    print(totalParams, totalParams2)
    print(op1.shape, op2.shape)
