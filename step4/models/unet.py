import torch
import torch.nn as nn


class DoubleConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		"""DoubleConv class initializes a double convolution block

		args:
			in_channels: number of input channels
			out_channels: number of output channels
		"""
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True))

	def forward(self, x):
		"""forward function performs forward pass through double convolution block

		args:
			x: input tensor

		returns:
			output tensor after double convolution
		"""
		return self.seq(x)


class UNet(nn.Module):
	def __init__(self, in_channels: int = 1, base: int = 32):
		"""UNet class initializes UNet model

		args:
			in_channels: number of input channels, default: 1
			base: base number of channels for first layer, default: 32
		"""
		super().__init__()
		self.d1 = DoubleConv(in_channels, base)
		self.p1 = nn.MaxPool2d(2)
		self.d2 = DoubleConv(base, base * 2)
		self.p2 = nn.MaxPool2d(2)
		self.d3 = DoubleConv(base * 2, base * 4)
		self.p3 = nn.MaxPool2d(2)
		self.d4 = DoubleConv(base * 4, base * 8)
		self.p4 = nn.MaxPool2d(2)
		self.center = DoubleConv(base * 8, base * 16)

		self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
		self.c4 = DoubleConv(base * 16, base * 8)
		self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
		self.c3 = DoubleConv(base * 8, base * 4)
		self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
		self.c2 = DoubleConv(base * 4, base * 2)
		self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
		self.c1 = DoubleConv(base * 2, base)
		self.out = nn.Conv2d(base, 1, 1)

	def forward(self, x):
		"""forward function performs forward pass through UNet

		args:
			x: input tensor of shape

		returns:
			output tensor of shape with segmentation mask
		"""
		x1 = self.d1(x)
		x2 = self.d2(self.p1(x1))
		x3 = self.d3(self.p2(x2))
		x4 = self.d4(self.p3(x3))
		x5 = self.center(self.p4(x4))
		x = self.u4(x5)
		x = torch.cat([x, x4], dim=1)
		x = self.c4(x)
		x = self.u3(x)
		x = torch.cat([x, x3], dim=1)
		x = self.c3(x)
		x = self.u2(x)
		x = torch.cat([x, x2], dim=1)
		x = self.c2(x)
		x = self.u1(x)
		x = torch.cat([x, x1], dim=1)
		x = self.c1(x)
		return self.out(x)
