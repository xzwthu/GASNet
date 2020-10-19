import torch.nn as nn
import torch

class UNet3D(nn.Module):
	def __init__(self, in_channels=1, n_classes=2, base_n_filter = 20):
		super(UNet3D, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU(inplace=True)
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='trilinear')
		self.softmax = nn.Softmax(dim=1)
		self.insnorm = nn.InstanceNorm3d(base_n_filter)


		self.conv = nn.Conv3d(in_channels,base_n_filter,3,1,1)
		self.conv1 = self.conv_norm_lrelu(base_n_filter,base_n_filter)
		self.max_cov2 = self.maxpool_cov_lrelu(base_n_filter,base_n_filter*2)
		self.cov2 = self.conv_norm_lrelu(base_n_filter*2,base_n_filter*2)
		self.max_cov3 = self.maxpool_cov_lrelu(base_n_filter*2,base_n_filter*4)
		self.cov3 = self.conv_norm_lrelu(base_n_filter*4,base_n_filter*4)
		self.max_cov4 = self.maxpool_cov_lrelu(base_n_filter*4,base_n_filter*8)
		self.cov4 = self.conv_norm_lrelu(base_n_filter*8,base_n_filter*8)

		self.up_cov3 = self.upscale_conv_norm_lrelu(base_n_filter*8,base_n_filter*4)
		self.de_cov3 = self.conv_norm_lrelu(base_n_filter*8,base_n_filter*4)
		self.up_cov2 = self.upscale_conv_norm_lrelu(base_n_filter*4,base_n_filter*2)
		self.de_cov2 = self.conv_norm_lrelu(base_n_filter*4,base_n_filter*2)
		self.up_cov1 = self.upscale_conv_norm_lrelu(base_n_filter*2,base_n_filter)
		self.de_cov1 = self.conv_norm_lrelu(base_n_filter*2,base_n_filter)

		self.pred = nn.Conv3d(base_n_filter,n_classes,3,1,1)



	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.Conv3d(feat_out,feat_out, kernel_size=(3,1,1), stride=1, padding=(1,0,0),bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True),
			nn.Conv3d(feat_out,feat_out,kernel_size=(1,3,3), stride=1,padding=(0,1,1),bias = False),
			nn.Conv3d(feat_out,feat_out,kernel_size=(3,1,1), stride=1,padding=(1,0,0),bias = False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True)
		)
	
	def maxpool_cov_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=2, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True)
		)
	def upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Upsample(scale_factor=2, mode='trilinear'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True)
		)


	def forward(self, x):
		x = self.lrelu(self.insnorm(self.conv(x)))
		x_cov1 = self.conv1(x)
		x_cov2 = self.max_cov2(x_cov1)
		x_cov2 = self.cov2(x_cov2)
		x_cov3 = self.max_cov3(x_cov2)
		x_cov3 = self.cov3(x_cov3)


		# x_cov4 = self.max_cov4(x_cov3)
		# x_cov4 = self.cov4(x_cov4)
		# x = self.up_cov3(x_cov4)
		# x = torch.cat([x,x_cov3],dim=1)
		# x = self.de_cov3(x)

		
		x = self.up_cov2(x_cov3)
		x = torch.cat([x,x_cov2],dim = 1)
		x = self.de_cov2(x)
		x = self.up_cov1(x)
		x = torch.cat([x,x_cov1],dim=1)
		x = self.de_cov1(x)
		out = self.pred(x)

		return out