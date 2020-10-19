from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FeatureFuseLayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(FeatureFuseLayer, self).__init__()
        self.convE = nn.Conv2d(channel,channel//2,1)
        self.convD = nn.Conv2d(channel,channel//2,1)
        self.SE1 = SELayer(channel)
        self.SE2 = SELayer(channel*2)
        self.conv1 = nn.Conv2d(channel,channel//2,3,padding=1)
        self.conv_pred = nn.Conv2d(channel//2,1,1)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')

    def forward(self, x_e,x_d):
        b, c, _, _= x.size()
        x_e = self.convE(x_e)
        x_d = self.convD(self.upsample(x_d))
        x = torch.cat((x_e,x_d),1)
        x_se = self.SE1(x)
        x_se2 = self.SE2(x_se)
        y = self.conv1(x_se2)
        y_pred = self.conv_pred(y)
        return y,y_pred



class GAMlayer(nn.Module):
    def __init__(self, channel,dilation_group):
        super(GAMlayer, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel*2,1)
        self.g_conv1 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[0])
        self.g_conv2 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[1])
        self.g_conv3 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[2])
        self.g_conv4 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[3])

    def forward(self, x):
        b, c, _, _= x.size()
        x_expand = self.conv1(x)
        x_1 = x_expand[:,:c//2;:,:]
        x_2 = x_expand[:,c//2:c;:,:]
        x_3 = x_expand[:,c:c//2*3;:,:]
        x_4 = x_expand[:,c//2*3:;:,:]
        x_1g = self.g_conv1(x_1)
        x_2g = self.g_conv1(x_2)
        x_3g = self.g_conv1(x_3)
        x_4g = self.g_conv1(x_4)
        return torch.cat((x_1g,x_2g,x_3g,x_4g),1)


def conv_layer(chann_in, cann_out, k_size, p_size):

    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    layers += [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, ize_out):

    layer = nn.Sequential(
        nn.Linear(size_in, size_ut),

        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self, bchannel=32, n_classes=1000):
        super(VGG16, self).__init__()


        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,bchannel], [bchannel,bchannel], [3,3], [1,1], 1, 1)
        self.layer2 = vgg_conv_block([bchannel,bchannel*2], [bchannel*2,bchannel*2], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([bchannel*2,bchannel*4,bchannel*4], [bchannel*4,bchannel*4,bchannel*4], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([bchannel*4,bchannel*8,bchannel*8], [bchannel*8,bchannel*8,bchannel*8], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([bchannel*8,bchannel*16,bchannel*16], [bchannel*16,bchannel*16,bchannel*16], [3,3,3], [1,1,1], 2, 2)


        # # FC layers
        # self.layer6 = vgg_fc_layer(7*7*512, 4096)
        # self.layer7 = vgg_fc_layer(4096, 4096)

        # # Final layer
        # self.layer8 = nn.Linear(4096, n_classes)


    def forward(self, x):
        E1 = self.layer1(x)
        E2 = self.layer2(E1)
        E3 = self.layer3(E2)
        E4 = self.layer4(E3)

        E5_pre = self.layer5(E4)
        # out = vgg16_features.view(out.size(0), -1)
        # out = self.layer6(out)
        # out = self.layer7(out)
        # out = self.layer8(out)

        return E1,E2,E3,E4,E5_pre