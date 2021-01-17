import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchgeometry as tgm

import matplotlib.pyplot as plt
from lib.utils import imshow_image
from sys import exit

from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
import numpy as np

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

		if use_cuda:
			self.model = self.model.cuda()

	def forward(self, batch):
		output = self.model(batch)
		o_size = output.size()
		output = output.view(o_size[0],o_size[1]*o_size[2],o_size[3],o_size[4])
		return output


	
