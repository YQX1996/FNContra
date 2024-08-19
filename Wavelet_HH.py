import random
import numpy as np
import torch
import torch.nn as nn
seq = nn.Sequential

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class generated_neg_D(nn.Module):
    def __init__(self, in_channels):
        super(generated_neg_D, self).__init__()
        self.WavePool = WavePool(in_channels)
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, pool=False)

    def forward(self, x_org):
        LL1, LH1, HL1, HH1 = self.WavePool(x_org)   ### 128
        LL2, LH2, HL2, HH2 = self.WavePool(LL1)     ### 64
        LL3, LH3, HL3, HH3 = self.WavePool(LL2)     ### 32
        LL4, LH4, HL4, HH4 = self.WavePool(LL3)     ### 16
        LL5, LH5, HL5, HH5 = self.WavePool(LL4)     ### 8

        x_easy_level5 = self.LL(LL5)
        x_easy_level4 = self.LL(x_easy_level5)
        x_easy_level3 = self.LL(x_easy_level4)
        x_easy_level2 = self.LL(x_easy_level3)
        x_easy_level1 = self.LL(x_easy_level2)
        x_easy = x_easy_level1

        x_middle1_level5 = self.LL(LL5) + self.LH(LH5)
        x_middle1_level4 = self.LL(x_middle1_level5) + self.LH(LH4)
        x_middle1_level3 = self.LL(x_middle1_level4) + self.LH(LH3)
        x_middle1_level2 = self.LL(x_middle1_level3) + self.LH(LH2)
        x_middle1_level1 = self.LL(x_middle1_level2) + self.LH(LH1)
        x_middle1 = x_middle1_level1

        x_middle2_level5 = self.LL(LL5) + self.HL(HL5)
        x_middle2_level4 = self.LL(x_middle2_level5) + self.HL(HL4)
        x_middle2_level3 = self.LL(x_middle2_level4) + self.HL(HL3)
        x_middle2_level2 = self.LL(x_middle2_level3) + self.HL(HL2)
        x_middle2_level1 = self.LL(x_middle2_level2) + self.HL(HL1)
        x_middle2 = x_middle2_level1

        x_middle3_level5 = self.LL(LL5) + self.HH(HH5)
        x_middle3_level4 = self.LL(x_middle3_level5) + self.HH(HH4)
        x_middle3_level3 = self.LL(x_middle3_level4) + self.HH(HH3)
        x_middle3_level2 = self.LL(x_middle3_level3) + self.HH(HH2)
        x_middle3_level1 = self.LL(x_middle3_level2) + self.HH(HH1)
        x_middle3 = x_middle3_level1
        x_middle = random.choice([x_middle1, x_middle2, x_middle3])

        x_hard1_level5 = self.LL(LL5) + self.LH(LH5) + self.HL(HL5)
        x_hard1_level4 = self.LL(x_hard1_level5) + self.LH(LH4) + self.HL(HL4)
        x_hard1_level3 = self.LL(x_hard1_level4) + self.LH(LH3) + self.HL(HL3)
        x_hard1_level2 = self.LL(x_hard1_level3) + self.LH(LH2) + self.HL(HL2)
        x_hard1_level1 = self.LL(x_hard1_level2) + self.LH(LH1) + self.HL(HL1)
        x_hard1 = x_hard1_level1

        x_hard2_level5 = self.LL(LL5) + self.LH(LH5) + self.HH(HH5)
        x_hard2_level4 = self.LL(x_hard2_level5) + self.LH(LH4) + self.HH(HH4)
        x_hard2_level3 = self.LL(x_hard2_level4) + self.LH(LH3) + self.HH(HH3)
        x_hard2_level2 = self.LL(x_hard2_level3) + self.LH(LH2) + self.HH(HH2)
        x_hard2_level1 = self.LL(x_hard2_level2) + self.LH(LH1) + self.HH(HH1)
        x_hard2 = x_hard2_level1

        x_hard3_level5 = self.LL(LL5) + self.HL(HL5) + self.HH(HH5)
        x_hard3_level4 = self.LL(x_hard3_level5) + self.HL(HL4) + self.HH(HH4)
        x_hard3_level3 = self.LL(x_hard3_level4) + self.HL(HL3) + self.HH(HH3)
        x_hard3_level2 = self.LL(x_hard3_level3) + self.HL(HL2) + self.HH(HH2)
        x_hard3_level1 = self.LL(x_hard3_level2) + self.HL(HL1) + self.HH(HH1)
        x_hard3 = x_hard3_level1
        x_hard = random.choice([x_hard1, x_hard2, x_hard3])

        return x_easy, x_middle, x_hard

class WavePool_HH(nn.Module):
    def __init__(self, in_channels):
        super(WavePool_HH, self).__init__()
        self.WavePool = WavePool(in_channels)
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, pool=False)

    def forward(self, x):
        LL, LH, HL, HH = self.WavePool(x)
        out = self.LH(LH) + self.HL(HL) + self.HH(HH)
        return out