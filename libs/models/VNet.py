import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
def ContBatchNorm3d(nchanl):
    return nn.InstanceNorm3d(nchanl)
# class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
#     # def _check_input_dim(self, input):
#     #     if input.dim() != 5:
#     #         raise ValueError('expected 5D input (got {}D input)'
#     #                          .format(input.dim()))
#     #     super(ContBatchNorm3d, self)._check_input_dim(input)
    
#     def forward(self, input):
#         # self._check_input_dim(input)
#         return F.batch_norm(
#             input, self.running_mean, self.running_var, self.weight, self.bias,
#             True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        # x16 = torch.cat((x, x, x, x, x, x, x, x,
        #                  x, x, x, x, x, x, x, x), 0)
        out = self.relu1(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=3, stride=2,padding=1)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.upsacle = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv = nn.Conv3d(inChans,outChans//2,kernel_size = 3,padding = 1)
        # self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.conv(self.upsacle(out))))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(inChans, 1, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        self.sigmoid = nn.Sigmoid()
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        # out = self.relu1(self.bn1(self.conv1(x)))
        out = self.sigmoid(self.conv2(x))

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        basic_channel = 16
        self.in_tr = InputTransition(basic_channel, elu)
        self.down_tr32 = DownTransition(basic_channel, 1, elu)
        self.down_tr64 = DownTransition(basic_channel*2, 2, elu)
        self.down_tr128 = DownTransition(basic_channel*4, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(basic_channel*8, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(basic_channel*16, basic_channel*16, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(basic_channel*8, basic_channel*8, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(basic_channel*8, basic_channel*4, 1, elu)
        self.up_tr32 = UpTransition(basic_channel*4, basic_channel*2, 1, elu)
        self.out_tr = OutputTransition(basic_channel*2, elu, nll)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        # print(out16.shape)
        # print(out32.shape)
        # print(out64.shape)
        # print(out128.shape)
        # out256 = self.down_tr256(out128)
        # out = self.up_tr256(out256, out128)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out