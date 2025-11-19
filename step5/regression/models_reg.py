import torch.nn as nn
from torchvision import models


class ResNetRegBin(nn.Module):
	def __init__(self, pretrained: bool = True):
		"""ResNetRegBin constructor initializes ResNet18-based regression and classification model

		args:
			pretrained: whether to use ImageNet pretrained weights
		"""
		super().__init__()
		self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
		self.backbone.fc = nn.Identity()
		self.regression = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
		self.classification = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

	def forward(self, x):
		"""forward function performs regression and classification on input images

		args:
			x: input img tensor of shape

		returns:
			tuple of diameter_predictions, classification_logits
		"""
		if x.size(1) == 1:
			x = x.repeat(1,3,1,1)
		f = self.backbone(x)
		diameter = self.regression(f).squeeze(1)
		logits = self.classification(f)
		return diameter, logits
