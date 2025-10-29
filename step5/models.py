import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models


class ImgEncoder(nn.Module):
	def __init__(self, out_dim: int = 128, pretrained: bool = False):
		super().__init__()
		self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
		self.backbone.fc = nn.Identity()
		self.proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, out_dim))

	def forward(self, x):
		# replicate gray to 3 channels to fully leverage pretrained conv1
		if x.size(1) == 1:
			x = x.repeat(1,3,1,1)
		ftrs = self.backbone(x)
		z = f.normalize(self.proj(ftrs), dim=-1)
		return z


class TxtEncoder(nn.Module):
	def __init__(self, vocab: dict = None, emb_dim: int = 128, out_dim: int = 128):
		super().__init__()
		self.vocab_size = max(vocab.values()) + 1 if vocab else 1000
		self.emb = nn.Embedding(self.vocab_size, emb_dim)
		self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
		self.proj = nn.Sequential(nn.Linear(emb_dim*2, 256), nn.ReLU(inplace=True), nn.Linear(256, out_dim))

	def forward(self, ids, lengths):
		e = self.emb(ids)
		packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
		out, _ = self.gru(packed)
		out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
		mask = (torch.arange(out.size(1), device=lengths.device)[None,:] < lengths[:,None]).float()
		feat = (out * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
		z = f.normalize(self.proj(feat), dim=-1)
		return z


def clip_loss(img_z, txt_z, temperature: float = 0.05):
	img_z = f.normalize(img_z, dim=-1)
	txt_z = f.normalize(txt_z, dim=-1)
	logits = img_z @ txt_z.t() / temperature
	target = torch.arange(img_z.size(0), device=img_z.device)
	li = f.cross_entropy(logits, target)
	lt = f.cross_entropy(logits.t(), target)
	return (li + lt) * 0.5
