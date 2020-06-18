import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.utils import save_image


class ResidualBlock(nn.Module):
	"""Residual Block with instance normalization."""
	def __init__(self, dim_in, dim_out):
		super(ResidualBlock, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

	def forward(self, x):
		return x + self.main(x)

class Generator1(nn.Module):
	"""Generator network."""
	def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
		super(Generator1, self).__init__()

		self.start_layers, self.encode_layers, self.resid_layers, self.decode_layers, self.final_layers = [], [], [], [], []
		self.start_layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
		self.start_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
		self.start_layers.append(nn.ReLU(inplace=True))

		# Down-sampling layers.
		curr_dim = conv_dim
		for i in range(2):
			self.encode_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
			self.encode_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
			self.encode_layers.append(nn.ReLU(inplace=True))
			curr_dim = curr_dim * 2

		# Bottleneck layers.
		for i in range(repeat_num):
			self.resid_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

		# Up-sampling layers.
		for i in range(2):
			self.decode_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
			self.decode_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
			self.decode_layers.append(nn.ReLU(inplace=True))
			curr_dim = curr_dim // 2

		self.final_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
		self.final_layers.append(nn.Tanh())

		self.start = nn.Sequential(*self.start_layers)
		self.encode = nn.Sequential(*self.encode_layers)
		self.resid = nn.Sequential(*self.resid_layers)
		self.decode = nn.Sequential(*self.decode_layers)
		self.final = nn.Sequential(*self.final_layers)
		# self.main = nn.Sequential(*layers)
		self.visualization = {}
   
   # def hook_fn(m, i, o):
   #     self.visualization[m] = o 

	def forward(self, x, c):
		# Replicate spatially and concatenate domain information.
		# Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
		# This is because instance normalization ignores the shifting (or bias) effect.
		c = c.view(c.size(0), c.size(1), 1, 1)
		c = c.repeat(1, 1, x.size(2), x.size(3))
		x = torch.cat([x, c], dim=1)

		x = self.start(x)
		x = self.encode(x)
		x = self.resid(x)
		x = self.decode(x)
		x = self.final(x)
		#for name, layer in self._modules.items():
		#    import pdb; pdb.set_trace()
		#    layer.register_forward_hook(hook_fn)
		return out



class Generator(nn.Module):
	"""Generator network."""
	def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
		super(Generator, self).__init__()

		layers = []
		layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
		layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
		layers.append(nn.ReLU(inplace=True))

		# Down-sampling layers.
		curr_dim = conv_dim
		for i in range(2):
			layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
			layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
			layers.append(nn.ReLU(inplace=True))
			curr_dim = curr_dim * 2

		# Bottleneck layers.
		for i in range(repeat_num):
			layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

		# Up-sampling layers.
		for i in range(2):
			layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
			layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
			layers.append(nn.ReLU(inplace=True))
			curr_dim = curr_dim // 2

		layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
		layers.append(nn.Tanh())
		self.main = nn.Sequential(*layers)
		self.visualization = {}
   
	def forward(self, x, c):
		# Replicate spatially and concatenate domain information.
		# Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
		# This is because instance normalization ignores the shifting (or bias) effect.
		import pdb; pdb.set_trace()
		c = c.view(c.size(0), c.size(1), 1, 1)
		c = c.repeat(1, 1, x.size(2), x.size(3))
		x = torch.cat([x, c], dim=1)
		out = self.main(x)
		return out


class Discriminator(nn.Module):
	"""Discriminator network with PatchGAN."""
	def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
		super(Discriminator, self).__init__()
		layers = []
		layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
		layers.append(nn.LeakyReLU(0.01))

		curr_dim = conv_dim
		for i in range(1, repeat_num):
			layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
			layers.append(nn.LeakyReLU(0.01))
			curr_dim = curr_dim * 2

		kernel_size = int(image_size / np.power(2, repeat_num))
		self.main = nn.Sequential(*layers)
		self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
		
	def forward(self, x):
		h = self.main(x)
		out_src = self.conv1(h)
		out_cls = self.conv2(h)
		return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
