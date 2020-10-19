import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# from Se_module import VGG16,SELayer,FeatureFuseLayer,GAMlayer

# BATCH_SIZE = 10
# LEARNING_RATE = 0.01
# EPOCH = 50
# N_CLASSES = 25


# transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()

#     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
#                        std  = [ 0.229, 0.224, 0.225 ]),
#     ])

# trainData = dsets.ImageFolder('../data/imagenet/train', transfom)
# testData = dsets.ImageFolder('../data/imagenet/test', transform)
# trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=Tru)
# testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


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
        self.SE2 = SELayer(channel)
        self.conv1 = nn.Conv2d(channel,channel//2,3,padding=1)
        self.conv_pred = nn.Conv2d(channel//2,1,1)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')

    def forward(self, x_e,x_d):
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
        self.g_conv1 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[0],padding=dilation_group[0])
        self.g_conv2 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[1],padding=dilation_group[1])
        self.g_conv3 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[2],padding=dilation_group[2])
        self.g_conv4 = nn.Conv2d(channel//2,channel//2,3,dilation=dilation_group[3],padding=dilation_group[3])
        self.conv2 = nn.Conv2d(channel*2,channel,1)
        self.conv3 = nn.Conv2d(channel,channel,3,padding=1)

    def forward(self, x):
        b, c, _, _= x.size()
        x_expand = self.conv1(x)
        x_1 = x_expand[:,:c//2,:,:]
        x_2 = x_expand[:,c//2:c,:,:]
        x_3 = x_expand[:,c:c//2*3,:,:]
        x_4 = x_expand[:,c//2*3:,:,:]
        x_1g = self.g_conv1(x_1)
        x_2g = self.g_conv1(x_2)
        x_3g = self.g_conv1(x_3)
        x_4g = self.g_conv1(x_4)
        x_cat = torch.cat((x_1g,x_2g,x_3g,x_4g),1)
        y = self.conv3(self.conv2(x_cat))
        return y


def conv_layer(chann_in, chann_out, k_size, p_size):

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
        self.layer1 = vgg_conv_block([1,bchannel], [bchannel,bchannel], [3,3], [1,1], 1, 1)
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

class JASNet(nn.Module):
    def __init__(self, reduction=4):
        super(JASNet, self).__init__()
        basechannel = 32
        self.VGG = VGG16(bchannel=basechannel)
        self.GAM1 = GAMlayer(basechannel*16,[1,3,6,9])
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.GAM2 = GAMlayer(basechannel*16,[1,2,3,4])
        self.FFL5 = FeatureFuseLayer(basechannel*16)
        self.FFL4 = FeatureFuseLayer(basechannel*8)
        self.FFL3 = FeatureFuseLayer(basechannel*4)
        self.FFL2 = FeatureFuseLayer(basechannel*2)
        self.FFL1 = FeatureFuseLayer(basechannel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        E1,E2,E3,E4,E5_pre = self.VGG(x)
        E5 = self.GAM1(E5_pre)
        E6 = self.GAM2(self.maxpool(E5))
        D5,p5 = self.FFL5(E5,E6)
        D4,p4 = self.FFL4(E4,D5)
        D3,p3 = self.FFL3(E3,D4)
        D2,p2 = self.FFL2(E2,D3)
        D1,p1 = self.FFL1(E1,D2)
        p1 = self.sigmoid(p1)
        p2 = self.sigmoid(p2)
        p3 = self.sigmoid(p3)
        p4 = self.sigmoid(p4)
        p5 = self.sigmoid(p5)
        return p5,p4,p3,p2,p1
