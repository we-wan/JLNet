import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models



class BasicConv2d(nn.Module):
    """
    basic convoluation model
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class res_unit(nn.Module):
    """
    this is the attention module before Residual structure
    """
    def __init__(self,channel,up_size = None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(res_unit,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = nn.Conv2d(channel,channel,3,padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size,up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        identity = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = torch.mul(identity,x)
        return x


class basenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _attenNet()
        self.fea = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 9 * 9, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 128),
        )

    def forward(self,x):
        x = self.base(x)
        x = x.view(-1, 9 * 9 * 128)
        x = self.fea(x)
        return x


class _atten(nn.Module):
    """
    the attention Module in <Learning part-aware attention networks for kinship verification>
    """
    def __init__(self):
        super(_atten,self).__init__()
        self.conv1 = nn.Conv2d(6,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.at1 = res_unit(32)
        self.at2 = res_unit(64)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        # self.fc1 = nn.Linear((9*9*128),512)
        # self.dp  = nn.Dropout()
        # self.fc2 = nn.Linear(512,2)

    def forward(self,x):
        """
        :param x: 6x64x64
        :return:
        """
        x = self.conv1(x)
        identity1 = x
        x = self.at1(x)
        x = identity1+x
        x = self.bn1(x)
        x = self.pool(F.relu(x))

        x = self.conv2(x)
        identity2 = x
        x = self.at2(x)
        x = identity2 + x
        x = self.bn2(x)
        x = self.pool(F.relu((x)))


        # x = x.view(-1, 9*9*128)
        # x = F.relu(self.fc1(x))
        # x = self.dp(x)
        # x = self.fc2(x)
        return x


class _attenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _atten()
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.at3 = res_unit(128,up_size=9)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self,x):
        x = self.base(x)
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        return x


class res_addNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.at3 = res_unit(128, up_size=9)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        return x

class each_brach(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.at3 = res_unit(128, up_size=9)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Sequential(
            nn.Linear((9 * 9 * 128), 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 2)
        )
    def forward(self, x):
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1, 9*9*128)
        x = self.fc(x)
        return x



class JLNet_basic(nn.Module):
    """
    concatenate 4x2 output + add loss layer
    """
    def __init__(self):
        super().__init__()
        self.base = _atten()

        self.fd_fc = each_brach()

        self.fs_fc = each_brach()

        self.md_fc = each_brach()

        self.ms_fc = each_brach()

    def fd_forward(self, x):
        x = self.base(x)
        x = self.fd_fc(x)
        return x

    def fs_forward(self, x):
        x = self.base(x)
        x = self.fs_fc(x)
        return x

    def md_forward(self, x):
        x = self.base(x)
        x = self.md_fc(x)
        return x

    def ms_forward(self, x):
        x = self.base(x)
        x = self.ms_fc(x)
        return x

    def forward(self, x):

        fd = self.fd_forward(x)
        fs = self.fs_forward(x)
        md = self.md_forward(x)
        ms = self.ms_forward(x)

        neg = torch.unsqueeze(torch.min(torch.cat((fd[:, 0:1], fs[:, 0:1], md[:, 0:1], ms[:, 0:1]), dim=1),dim=1)[0],
                              dim=1)
        all = torch.cat((neg, fd[:, 1:2], fs[:, 1:2], md[:, 1:2], ms[:, 1:2]), dim=1)

        return fd, fs, md, ms, all



