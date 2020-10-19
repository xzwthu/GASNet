
from functools import partial
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F 


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur,self).__init__()
        self.cov = nn.Conv3d(1,1,5,1,2)
        kk = torch.Tensor(np.ones([1,1,5,5,5]))/125
        self.cov.weight=torch.nn.Parameter(kk)
    def forward(self,x):
        x = self.cov(x)
        return x


class Laplace(nn.Module):
    def __init__(self):
        super(Laplace,self).__init__()
        self.cov_x = nn.Conv3d(1,1,2,1,1)
        self.cov_y = nn.Conv3d(1,1,2,1,1)
        self.cov_z = nn.Conv3d(1,1,2,1,1)
        kk = torch.Tensor(np.ones([1,1,2,2,2]))
        kk[0,0,0,:,:] = -1
        self.cov_x.weight=torch.nn.Parameter(kk)
        self.cov_y.weight=torch.nn.Parameter(kk.transpose(2,3))
        self.cov_z.weight=torch.nn.Parameter(kk.transpose(2,4))
    def forward(self,x):
        x_x = self.cov_x(x)
        x_y = self.cov_x(x)
        x_z = self.cov_x(x)
        return (x_x**2+x_y**2+x_z**2).mean()

class Discriminator(nn.Module):
    def __init__(self, input_channel=1,basic_channel=32,classes = 1):
        super(Discriminator,self).__init__()
        # block_num = 300
        self.input_channel = input_channel
        self.basic_channel = basic_channel
        self.layer_num = basic_channel*16*2*5*5
        self.layer = nn.Linear(self.layer_num,classes)
        self.classes = classes
        self.cov_init = self.cir_init(input_channel,basic_channel) #(32,80,80)
        self.cov1 = self.cir_block(basic_channel)
        # self.cov2 = self.cir_block(basic_channel)
        self.cov3 = self.cir_down_block(basic_channel,basic_channel*2) #(16,40,40)
        # self.cov4 = self.cir_block(basic_channel*2)
        self.cov5 = self.cir_block(basic_channel*2)
        self.cov6 = self.cir_down_block(basic_channel*2,basic_channel*4) #(32,80,80)/4
        # self.cov7 = self.cir_block(basic_channel*4)
        self.cov8 = self.cir_block(basic_channel*4)
        self.cov9 = self.cir_down_block(basic_channel*4,basic_channel*8) #(32,80,80)/8
        # self.cov10 = self.cir_block(basic_channel*8)
        self.cov11 = self.cir_block(basic_channel*8)
        self.cov12 = self.cir_down_block(basic_channel*8,basic_channel*16) #(32,80,80)/16
        self.cov13 = self.cir_block(basic_channel*16)
        self.out_conv = nn.Conv3d(basic_channel*16,1,1,1,0)
        self.flat_conv2 = nn.Sequential(
            nn.Conv3d(basic_channel*16,1,1,1,0),
            nn.InstanceNorm3d(1),
        )
        self.att = self.attention_block(basic_channel*4)
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_init = self.cov_init(x)
        x_cov1 = self.cov1(x_init)
        x_cov3 = self.cov3(x_cov1)
        x_cov5 = self.cov5(x_cov3)
        x_cov6 = self.cov6(x_cov5)
        x_att = self.att(x_cov6)
        # x_cov6 = x_cov6*(x_att)
        x_cov8 = self.cov8(x_cov6)
        x_cov9 = self.cov9(x_cov8)
        x_cov11 = self.cov11(x_cov9)
        x_cov12 = self.cov12(x_cov11)
        x_flatten = x_flat.view(x_cov12.size(0),-1)
        out1 = self.layer(x_flatten)
        # out2 = self.out_conv(x_cov12)
        # import pdb; pdb.set_trace()
        return [out1],x_att
    def cir_block(self,basic_channel):
        return nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv3d(basic_channel,basic_channel,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))),
            nn.InstanceNorm3d(basic_channel),
            nn.LeakyReLU(0.1),
            # torch.nn.utils.spectral_norm(nn.Conv3d(basic_channel,basic_channel,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))),
            # nn.LeakyReLU(0.1),
        )
    def cir_down_block(self,input_channel,output_channel):
        return nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv3d(input_channel,output_channel,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1))),
            nn.InstanceNorm3d(output_channel),
            nn.LeakyReLU(0.1)
        )
    def cir_init(self,input_channel,output_channel):
        return nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv3d(input_channel,output_channel,kernel_size=(1,4,4),stride=(1,2,2),padding=(0,1,1))),
            nn.LeakyReLU(0.1,inplace=True)
        )
    def attention_block(self,f_in):
        return nn.Sequential(
            nn.Conv3d(f_in,1,kernel_size=1),
            nn.Sigmoid()
        )


class Discriminator2(nn.Module):
    def __init__(self, input_channel=1,basic_channel=64,classes = 1):
        super(Discriminator2,self).__init__()
        self.input_channel = input_channel
        self.basic_channel = basic_channel
        self.layer_num = basic_channel*16*2*5*5
        self.layer = nn.Linear(self.layer_num,classes)
        self.classes = classes
        self.cov_init = self.cir_init(input_channel,basic_channel)
        self.cov1 = self.cir_block(basic_channel)
        self.cov2 = self.cir_block(basic_channel)
        self.cov3 = self.cir_down_block(basic_channel,basic_channel*2)
        self.cov4 = self.cir_block(basic_channel*2)
        self.cov5 = self.cir_block(basic_channel*2)
        self.cov6 = self.cir_down_block(basic_channel*2,basic_channel*4)
        self.cov7 = self.cir_block(basic_channel*4)
        self.cov8 = self.cir_block(basic_channel*4)
        self.cov9 = self.cir_down_block(basic_channel*4,basic_channel*8)
        self.cov10 = self.cir_block(basic_channel*8)
        self.cov11 = self.cir_block(basic_channel*8)
        self.cov12 = self.cir_down_block(basic_channel*8,basic_channel*16)
        self.cov13 = self.cir_block(basic_channel*16)

        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_init = self.cov_init(x)
        x_cov1 = self.cov1(x_init)
        x_cov1 += x_init
        x_cov2 = self.cov2(x_cov1)
        x_cov2 += x_cov1
        x_cov3 = self.cov3(x_cov2)
        x_cov4 = self.cov4(x_cov3)
        x_cov4 += x_cov3
        x_cov5 = self.cov5(x_cov4)
        x_cov5 += x_cov4
        x_cov6 = self.cov6(x_cov5)
        x_cov7 = self.cov7(x_cov6)
        x_cov7 += x_cov6
        x_cov8 = self.cov8(x_cov7)
        x_cov8 += x_cov7
        x_cov9 = self.cov9(x_cov8)
        x_cov10 = self.cov10(x_cov9)
        x_cov10 += x_cov9
        x_cov11 = self.cov11(x_cov10)
        x_cov11 += x_cov10
        x_cov12 = self.cov12(x_cov11)
        x_cov13 = self.cov13(x_cov12)
        x_cov13 += x_cov12
        # import pdb; pdb.set_trace()
        x_flatten = x_cov13.view(x_cov13.size(0),-1)
        out = self.layer(x_flatten)
        out = self.sigmoid(out)
        return out,x_flatten
    def cir_block(self,basic_channel):
        return nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv3d(basic_channel,basic_channel,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))),
            torch.nn.utils.spectral_norm(nn.Conv3d(basic_channel,basic_channel,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0))),
            nn.LeakyReLU(0.1)
        )
    def cir_down_block(self,input_channel,output_channel):
        return nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv3d(input_channel,output_channel,kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))),
            torch.nn.utils.spectral_norm(nn.Conv3d(output_channel,output_channel,kernel_size=(3,1,1),stride=(2,1,1),padding=(1,0,0))),
            nn.LeakyReLU(0.1)
        )
    def cir_init(self,input_channel,output_channel):
        return nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv3d(input_channel,output_channel,kernel_size=(1,5,5),stride=(1,2,2),padding=(0,2,2))),
            nn.LeakyReLU(0.1,inplace=True)
        )
class Generator(nn.Module):
    def __init__(self, input_channel=1, basic_channel=64):
        super(Generator,self).__init__()
        self.input_channel = input_channel
        self.basic_channel = basic_channel
        self.conv = nn.Conv3d(input_channel,basic_channel,kernel_size=3,stride=1,padding=1)
        self.bn3d = nn.InstanceNorm3d(basic_channel)
        self.relu = nn.ReLU(inplace=True)
        conv1 = nn.Sequential()
        for i in range(2):
            conv1.add_module(str(1)+'_'+str(i)+'conv',nn.Conv3d(basic_channel,basic_channel,kernel_size=3,stride=1,padding=1))
            conv1.add_module(str(1)+'_'+str(i)+'bn',nn.InstanceNorm3d(basic_channel))
            conv1.add_module(str(1)+'_'+str(i)+'relu',nn.ReLU(inplace=True))
        conv2 = nn.Sequential()
        for i in range(2):
            conv2.add_module(str(2)+'_'+str(i)+'conv',nn.Conv3d(basic_channel,basic_channel,kernel_size=3,stride=1,padding=1))
            conv2.add_module(str(2)+'_'+str(i)+'bn',nn.InstanceNorm3d(basic_channel))
            conv2.add_module(str(2)+'_'+str(i)+'relu',nn.ReLU(inplace=True))
        conv3 = nn.Sequential()
        for i in range(2):
            conv3.add_module(str(3)+'_'+str(i)+'conv',nn.Conv3d(basic_channel,basic_channel,kernel_size=3,stride=1,padding=1))
            conv3.add_module(str(3)+'_'+str(i)+'bn',nn.InstanceNorm3d(basic_channel))
            conv3.add_module(str(3)+'_'+str(i)+'relu',nn.ReLU(inplace=True))
        out = nn.Sequential()
        out.add_module('output_conv',nn.Conv3d(basic_channel,input_channel,kernel_size=3,stride=1,padding=1))
        out.add_module('output_tanh',nn.Sigmoid())
        self.conv1_G = conv1
        self.conv2_G = conv2
        self.conv3_G = conv3
        self.out_G = out

    def forward(self, x):
        x = self.relu(self.bn3d(self.conv(x)))
        x_1 = self.conv1_G(x)
        x_1add =x+x_1
        x_2 = self.conv2_G(x_1add)
        x_2add =x+x_2
        # x_3 = self.conv3_G(x_2add)
        # x_3add =x+x_3
        out = self.out_G(x_2add)
        return out

class Generator3(nn.Module):
    def __init__(self, input_channel=2, basic_channel=32):
        super(Generator3,self).__init__()
        self.input_channel = input_channel
        self.basic_channel = basic_channel
        self.conv1 = self.cov_norm_relu(input_channel,basic_channel,(1,4,4),(1,2,2),(0,1,1))
        self.conv2 = self.cov_norm_relu(self.basic_channel,self.basic_channel*2)
        self.conv3 = self.cov_norm_relu(basic_channel*2,basic_channel*4)
        self.conv4 = self.cov_norm_relu(basic_channel*4,basic_channel*8)
        self.conv5 = self.cov_norm_relu(basic_channel*8,basic_channel*16)
        self.up_conv4 = self.upcov_norm_relu(basic_channel*16,basic_channel*8)
        self.up_conv3 = self.upcov_norm_relu(basic_channel*8*2,basic_channel*4)
        self.up_conv2 = self.upcov_norm_relu(basic_channel*4*2,basic_channel*2)
        self.up_conv1 = self.upcov_norm_relu(basic_channel*2*2,basic_channel)
        self.out_4 = self.conv_tan(basic_channel*8,1)
        self.out_3 = self.conv_tan(basic_channel*4,1)
        self.out_2 = self.conv_tan(basic_channel*2,1)
        self.tanh = nn.Tanh()
        self.up = nn.Upsample(scale_factor=(4,8,8),mode = 'trilinear')
        out = nn.Sequential()
        out.add_module('output_upscale',nn.Upsample(scale_factor=(1,2,2), mode='trilinear'))
        out.add_module('output_conv',nn.Conv3d(basic_channel*2,input_channel,kernel_size=3,stride=1,padding=1,bias=False))
        self.out_G = out
    def cov_norm_relu(self,input_channel,output_channel,kernel_size=3,stride=2,pad=1,LeakyRelu=True):
        if LeakyRelu:
            return nn.Sequential(
                nn.Conv3d(input_channel,output_channel,kernel_size=kernel_size,stride=stride,padding=pad),
                nn.InstanceNorm3d(output_channel),
                nn.LeakyReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv3d(input_channel,output_channel,kernel_size=kernel_size,stride=stride,padding=padding),
                nn.InstanceNorm3d(output_channel),
                nn.ReLU(inplace=True)
            )
    def upcov_norm_relu(self,input_channel,output_channel,kernel_size=3,stride = 2,padding=1,LeakyRelu=False):
        if LeakyRelu:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(output_channel),
                nn.LeakyReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(output_channel),
                nn.ReLU(inplace=True)
            )
    def conv_tan(self, input_channel, output_channel):
        return nn.Sequential(
            nn.Conv3d(input_channel,output_channel,kernel_size=1),
        )
    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        d4_ = self.up_conv4(e5)
        d4 = torch.cat([d4_,e4],dim=1)
        d3_ = self.up_conv3(d4)
        d3 = torch.cat([d3_,e3],dim=1)
        d2_ = self.up_conv2(d3)
        d2 = torch.cat([d2_,e2],dim=1)
        d1_ = self.up_conv1(d2)
        d1 = torch.cat([d1_,e1],dim=1)
        # out4 = self.out_4(d4_)
        out3 = self.out_3(d3_)
        # out2 = self.out_2(d2_)
        out = self.out_G(d1)#+self.up(out3)
        # return self.tanh(out),out4,self.tanh(out3),out2
        return self.tanh(out)