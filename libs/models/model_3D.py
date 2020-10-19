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
		self.sigmoid = nn.Sigmoid()



	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.Conv3d(feat_out,feat_out, kernel_size=(3,1,1), stride=1, padding=(1,0,0),bias=False),
			# nn.GroupNorm(4,feat_out),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True),
			nn.Conv3d(feat_out,feat_out,kernel_size=(1,3,3), stride=1,padding=(0,1,1),bias = False),
			nn.Conv3d(feat_out,feat_out,kernel_size=(3,1,1), stride=1,padding=(1,0,0),bias = False),
			# nn.GroupNorm(4,feat_out),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True)
		)
	
	def maxpool_cov_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=2, padding=1, bias=False),
			# nn.GroupNorm(4,feat_out),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True)
		)
	def upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Upsample(scale_factor=2, mode='trilinear'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			# nn.GroupNorm(4,feat_out),
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
		out = self.sigmoid(self.pred(x))

		return out



class UNet3D_orig(nn.Module):
	def __init__(self, in_channels=1, n_classes=2, base_n_filter = 20):
		super(UNet3D_orig, self).__init__()
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
		self.sigmoid = nn.Sigmoid()



	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True),
			nn.Conv3d(feat_out,feat_out,kernel_size=(3,3,3), stride=1,padding=(1,1,1),bias = False),
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
		out = self.sigmoid(self.pred(x))

		return out



class UNet3D_shallow(nn.Module):
	def __init__(self, in_channels=1, n_classes=2, base_n_filter = 16):
		super(UNet3D_shallow, self).__init__()
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
		self.sigmoid = nn.Sigmoid()



	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.Conv3d(feat_out,feat_out, kernel_size=(3,1,1), stride=1, padding=(1,0,0),bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU(inplace=True),
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
		# x_cov3 = self.max_cov3(x_cov2)
		# x_cov3 = self.cov3(x_cov3)

		# # x_cov4 = self.max_cov4(x_cov3)
		# # x_cov4 = self.cov4(x_cov4)
		# # x = self.up_cov3(x_cov4)
		# # x = torch.cat([x,x_cov3],dim=1)
		# # x = self.de_cov3(x)

		# x = self.up_cov2(x_cov3)
		# x = torch.cat([x,x_cov2],dim = 1)
		# x = self.de_cov2(x)
		x = self.up_cov1(x_cov2)
		x = torch.cat([x,x_cov1],dim=1)
		x = self.de_cov1(x)
		out = self.sigmoid(self.pred(x))

		return out


class VGGBlock(nn.Module):
	def __init__(self, in_channels, middle_channels, out_channels):
		super().__init__()
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size= 3, padding=1)
		self.bn1 = nn.InstanceNorm3d(middle_channels)
		self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size= 3, padding=1)
		self.bn2 = nn.InstanceNorm3d(out_channels)

	def forward(self, x):
		out = self.conv1(x)

		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)


		return out

class NestedUNet(nn.Module):
	def __init__(self, num_classes=1, input_channels=1, deep_supervision=False, **kwargs):
		super().__init__()
		basic_channel = 12
		nb_filter = [basic_channel,basic_channel*2,basic_channel*4,basic_channel*8,basic_channel*16]

		self.deep_supervision = deep_supervision

		self.pool = nn.MaxPool3d(2,2)
		self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


		self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
		self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])

		self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
		self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
		self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

		self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
		self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

		self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
		self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

		self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
		self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

		self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
		self.sigmoid = nn.Sigmoid()
		if self.deep_supervision:
			self.final1 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
			self.final2 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
			self.final3 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
			self.final4 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
		else:
			self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)


	def forward(self, input):
		x0_0 = self.conv0_0(input)
		x1_0 = self.conv1_0(self.pool(x0_0))
		x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

		x2_0 = self.conv2_0(self.pool(x1_0))
		x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

		x3_0 = self.conv3_0(self.pool(x2_0))
		x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

		# x4_0 = self.conv4_0(self.pool(x3_0))
		# x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		# x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
		# x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
		# x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

		if self.deep_supervision:
			output1 = self.final1(x0_1)
			output2 = self.final2(x0_2)
			output3 = self.final3(x0_3)
			output4 = self.final4(x0_4)
			return [output1, output2, output3, output4]

		else:
			output = self.sigmoid(self.final(x0_3))
			return output