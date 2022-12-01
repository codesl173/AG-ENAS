import torch
from torch import FloatTensor
import torch.nn as nn
import torch.nn.functional as F


MyOps = {
    0: lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    1: lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    2: lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    3: lambda C, stride, affine: FRconv3(C, C, stride=1, affine=affine),
    4: lambda C, stride, affine: FRconv5(C, C, stride=1, affine=affine),
    5: lambda C, stride, affine: LBCNN(C, C, 3, stride, padding=1),
    6: lambda C, stride, affine: StdConv(C, C, 3, stride, 1, affine=affine),
    7: lambda C, stride, affine: FacConv(C, C, 3, stride, 1, affine=affine) if stride == 1 else FactorizedReduce(C, C, affine=affine),
    8: lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    9: lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine)
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding=(padding, 0), bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding=(0, padding), bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DS_layer3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DS_layer3, self).__init__()
        self.depth_conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, groups=in_planes, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.point_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        out = F.relu(self.bn1(self.depth_conv(input)))
        out = F.relu(self.bn2(self.point_conv(out)))
        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class DS_layer5(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DS_layer5, self).__init__()
        self.depth_conv = nn.Conv2d(in_planes, in_planes, kernel_size=5, groups=in_planes, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.point_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        out = F.relu(self.bn1(self.depth_conv(input)))
        out = F.relu(self.bn2(self.point_conv(out)))
        return out


class Conv_3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv_3, self).__init__()
        self.conv_3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        out = self.conv_3(input)
        out = F.relu(self.bn1(out))
        return out


class Conv_5(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv_5, self).__init__()
        self.conv_5 = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        out = self.conv_5(input)
        out = F.relu(self.bn1(out))
        return out


class Conv_1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv_1, self).__init__()
        self.conv_1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        out = self.conv_1(input)
        out = F.relu(self.bn1(out))
        return out


class Inception3(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Inception3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class Inception5(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Inception5, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(5, 1), padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class LBCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(LBCNN, self).__init__()
        self.nInputPlane = in_channels
        self.nOutputPlane = out_channels
        self.kW = kernel_size
        self.LBCNN = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.LBCNN.weight.requires_grad = False
        numElements = self.nInputPlane * self.nOutputPlane * self.kW * self.kW
        index = torch.randperm(numElements)
        self.LBCNN.weight.copy = (torch.Tensor(self.nOutputPlane, self.nInputPlane, self.kW, self.kW).uniform_(0, 1))
        temp = (torch.bernoulli(self.LBCNN.weight.copy) * 2 - 1).view(-1)
        for i in range(1, int(numElements / 2)):
            temp[index[i]] = 0
        self.LBCNN.weight.copy = temp.view(self.nOutputPlane, self.nInputPlane, self.kW, self.kW)
        self.LBCNN.weight = nn.Parameter(self.LBCNN.weight.copy)

    def forward(self, input):
        return self.LBCNN.forward(input)


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FRconv3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, affine=True):
        super(FRconv3, self).__init__()
        self.conv1 = DilConv(inplanes, planes, 3, stride, 2, 2, affine=affine)
        self.eca = eca_layer(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.eca(out)
        return out


class FRconv5(nn.Module):
    def __init__(self, inplanes, planes, stride=1, affine=True):
        super(FRconv5, self).__init__()
        self.conv1 = DilConv(inplanes, planes, 5, stride, 4, 2, affine=affine)
        self.eca = eca_layer(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.eca(out)
        return out


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
