import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchgeometry as tgm
import numpy as np

import matplotlib.pyplot as plt
from sys import exit
import math

from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces


class DenseFeatureExtractionModuleRotInv(nn.Module):
    def __init__(self, finetune_feature_extraction=False, use_cuda=False):
        super(DenseFeatureExtractionModuleRotInv, self).__init__()

        model = models.vgg16(pretrained=True)
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
            'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2',
            'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv3_3_idx = vgg16_layers.index('conv3_3')

        geometric_conv_channels = 512//8
        rot_inv_layers = [
            P4MConvZ2(256, geometric_conv_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            P4MConvP4M(geometric_conv_channels, geometric_conv_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            P4MConvP4M(geometric_conv_channels, geometric_conv_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        
        self.model = nn.Sequential(
            *list(model.features.children())[: conv3_3_idx + 2],
            *rot_inv_layers
        )

        self.num_channels = 512

        # Fix forward parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune_feature_extraction:
            # Unlock conv4_3
            for param in list(self.model.parameters())[-2 :]:
                param.requires_grad = True
        
    def forward(self, batch):
        output = self.model(batch)
        o_size = output.size()
        output = output.view(o_size[0],o_size[1]*o_size[2],o_size[3],o_size[4])
        return output



def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0
    
    N = gspace.fibergroup.order()
    
    if fixparams:
        planes *= math.sqrt(N)
    
    planes = planes / N
    planes = int(planes)
    
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a trivial feature map with the specified number of channels"""
    
    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())
        
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)

def conv5x5(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=2,
            dilation=1, bias=True):
    """5x5 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 5,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                    #   frequencies_cutoff=lambda r: 3*r,
                      )

def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=True):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                    #   frequencies_cutoff=lambda r: 3*r,
                      )

FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}

class DenseFeatureExtractionModuleE2Inv(nn.Module):
    def __init__(self):
        super(DenseFeatureExtractionModuleE2Inv, self).__init__()

        filters = np.array([32,32, 64,64, 128,128,128, 256,256,256, 512,512,512], dtype=np.int32)*2
        
        # number of rotations to consider for rotation invariance
        N = 8
        
        self.gspace = gspaces.Rot2dOnR2(N)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        ip_op_types = [
            self.input_type,
        ]
        
        for filter_ in filters[:10]:
            ip_op_types.append(FIELD_TYPE['regular'](self.gspace, filter_, fixparams=False))

        self.model = enn.SequentialModule(*[
            conv3x3(ip_op_types[0], ip_op_types[1]),
            enn.ReLU(ip_op_types[1], inplace=True),
            conv3x3(ip_op_types[1], ip_op_types[2]),
            enn.ReLU(ip_op_types[2], inplace=True),
            enn.PointwiseMaxPool(ip_op_types[2], 2),

            conv3x3(ip_op_types[2], ip_op_types[3]),
            enn.ReLU(ip_op_types[3], inplace=True),
            conv3x3(ip_op_types[3], ip_op_types[4]),
            enn.ReLU(ip_op_types[4], inplace=True),
            enn.PointwiseMaxPool(ip_op_types[4], 2),

            conv3x3(ip_op_types[4], ip_op_types[5]),
            enn.ReLU(ip_op_types[5], inplace=True),
            conv3x3(ip_op_types[5], ip_op_types[6]),
            enn.ReLU(ip_op_types[6], inplace=True),
            conv3x3(ip_op_types[6], ip_op_types[7]),
            enn.ReLU(ip_op_types[7], inplace=True),

            enn.PointwiseAvgPool(ip_op_types[7], 2),

            conv5x5(ip_op_types[7], ip_op_types[8]),
            enn.ReLU(ip_op_types[8], inplace=True),
            conv5x5(ip_op_types[8], ip_op_types[9]),
            enn.ReLU(ip_op_types[9], inplace=True),
            conv5x5(ip_op_types[9], ip_op_types[10]),
            enn.ReLU(ip_op_types[10], inplace=True),

            # conv3x3(ip_op_types[7], ip_op_types[8], padding=2, dilation=2),
            # enn.ReLU(ip_op_types[8], inplace=True),
            # conv3x3(ip_op_types[8], ip_op_types[9], padding=2, dilation=2),
            # enn.ReLU(ip_op_types[9], inplace=True),
            # conv3x3(ip_op_types[9], ip_op_types[10], padding=2, dilation=2),
            # enn.ReLU(ip_op_types[10], inplace=True),
            # enn.PointwiseMaxPool(ip_op_types[10], 2),

            # conv3x3(ip_op_types[10], ip_op_types[11]),
            # enn.ReLU(ip_op_types[11], inplace=True),
            # conv3x3(ip_op_types[11], ip_op_types[12]),
            # enn.ReLU(ip_op_types[12], inplace=True),
            # conv3x3(ip_op_types[12], ip_op_types[13]),
            # enn.ReLU(ip_op_types[13], inplace=True),

            enn.GroupPooling(ip_op_types[10])
        ])

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        out = self.model(x)
        out = out.tensor
        return out


if __name__=="__main__":
    test_model = DenseFeatureExtractionModuleE2Inv()
    ten = torch.randn(1,3,100,100)
    op = test_model(ten)
    print(op.shape)
