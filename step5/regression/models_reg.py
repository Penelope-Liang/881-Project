import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetRegBin(nn.Module):
	def __init__(self, pretrained: bool = True):
		super().__init__()
		self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
		self.backbone.fc = nn.Identity()
		self.reg_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
		self.cls_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

	def forward(self, x):
		if x.size(1) == 1:
			x = x.repeat(1,3,1,1)
		f = self.backbone(x)
		diam = self.reg_head(f).squeeze(1)
		logits = self.cls_head(f)
		return diam, logits


