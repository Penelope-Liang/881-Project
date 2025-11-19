import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models


class ImgEncoder(nn.Module):
	"""ImgEncoder class implements ResNet18-based img encoder for CLIP training

	args:
		out_dim: output embedding dimension
		pretrained: whether to use ImageNet pretrained weights
	"""
	def __init__(self, out_dim: int = 128, pretrained: bool = False):
		super().__init__()
		self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
		self.backbone.fc = nn.Identity()
		self.proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, out_dim))

	def forward(self, x):
		"""forward function encodes input images to normalized embeddings

		args:
			x: input img tensor of shape (batch, channels, height, width)

		returns:
			normalized embedding tensor of shape (batch, out_dim)
		"""
		if x.size(1) == 1:
			x = x.repeat(1,3,1,1)
		ftrs = self.backbone(x)
		z = f.normalize(self.proj(ftrs), dim=-1)
		return z


class TxtEncoder(nn.Module):
	"""TxtEncoder class implements bidirectional GRU-based text encoder for CLIP-style training

	args:
		vocab: vocab dict mapping tokens to IDs
		emb_dim: embedding dimension for tokens
		out_dim: output embedding dimension
	"""
	def __init__(self, vocab: dict = None, emb_dim: int = 128, out_dim: int = 128):
		super().__init__()
		self.vocab_size = max(vocab.values()) + 1 if vocab else 1000
		self.emb = nn.Embedding(self.vocab_size, emb_dim)
		self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
		self.proj = nn.Sequential(nn.Linear(emb_dim*2, 256), nn.ReLU(inplace=True), nn.Linear(256, out_dim))

	def forward(self, ids, lengths):
		"""forward function encodes input text sequences to normalized embeddings

		args:
			ids: token ID tensor of shape (batch, seq_len)
			lengths: actual sequence lengths tensor of shape (batch,)

		returns:
			normalized embedding tensor of shape (batch, out_dim)
		"""
		e = self.emb(ids)
		packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
		out, _ = self.gru(packed)
		out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
		mask = (torch.arange(out.size(1), device=lengths.device)[None,:] < lengths[:,None]).float()
		feat = (out * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
		z = f.normalize(self.proj(feat), dim=-1)
		return z


def clip_loss(img_z, txt_z, temperature: float = 0.05):
	"""clip_loss function computes symmetric CLIP contrastive loss between img and text embeddings

	args:
		img_z: normalized img embeddings of shape (batch, dim)
		txt_z: normalized text embeddings of shape (batch, dim)
		temperature: temperature scaling parameter

	returns:
		symmetric contrastive loss value
	"""
	img_z = f.normalize(img_z, dim=-1)
	txt_z = f.normalize(txt_z, dim=-1)
	logits = img_z @ txt_z.t() / temperature
	target = torch.arange(img_z.size(0), device=img_z.device)
	img_to_text = f.cross_entropy(logits, target)
	text_to_img = f.cross_entropy(logits.t(), target)
	return (img_to_text + text_to_img) * 0.5
