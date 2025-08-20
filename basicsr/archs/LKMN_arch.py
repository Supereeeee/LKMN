from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile  # 计算参数量和运算量
from basicsr.archs.arch_util import default_init_weights


class CA(nn.Module):
    def __init__(self, channels):
        super(CA, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.AdaptiveAvgPool(x))
        out = out * x
        return out


class PLKB(nn.Module):
    '''
    corresponding to Enhanced Partial Large Kernel Block (EPLKB) in paper
    '''

    def __init__(self, channels, large_kernel, split_group):
        super(PLKB, self).__init__()
        self.channels = channels
        self.split_group = split_group
        self.split_channels = int(channels // split_group)
        self.CA = CA(channels)
        self.DWConv_Kx1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(large_kernel, 1), stride=1,
                                    padding=(large_kernel // 2, 0), groups=self.split_channels)
        self.DWConv_1xK = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(1, large_kernel), stride=1,
                                    padding=(0, large_kernel // 2), groups=self.split_channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        # channel shuffle
        B, C, H, W = x.size()
        x = x.reshape(B, self.split_channels, self.split_group, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)

        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1)

        # channel attention
        x1 = self.CA(x1)

        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1))
        out = torch.cat((x1, x2), dim=1)
        out = self.act(self.conv1(out))
        return out


class HFAB(nn.Module):
    '''
    Hybrid Feature Aggregation Block (HFAB)
    '''

    def __init__(self, channels, large_kernel, split_group):
        super(HFAB, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_group)
        self.DWConv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.DWConv3(x)
        x2 = self.PLKB(x)
        out = self.act(self.conv1(torch.cat((x1, x2), dim=1)))
        return out


class HFDB(nn.Module):
    '''
    Hybrid Feature Distillation Block (HFDB)
    '''

    def __init__(self, channels, large_kernel, split_group):
        super(HFDB, self).__init__()
        self.c1_d = nn.Conv2d(channels, channels // 2, 1)
        self.c1_r = HFAB(channels, large_kernel, split_group)
        self.c2_d = nn.Conv2d(channels, channels // 2, 1)
        self.c2_r = HFAB(channels, large_kernel, split_group)
        self.c3_d = nn.Conv2d(channels, channels // 2, 1)
        self.c3_r = HFAB(channels, large_kernel, split_group)
        self.c4 = nn.Conv2d(channels, channels // 2, 1)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.act(self.c5(out))
        return out


class Scaler(nn.Module):
    def __init__(self, channels, init_value=1e-5, requires_grad=True):
        super(Scaler, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(1, channels, 1, 1),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class CGFN(nn.Module):
    '''
    Cross-Gate Feed-Forward Network (CGFN)
    '''
    def __init__(self, channels, large_kernel, split_group):
        super(CGFN, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_group)
        self.DWConv_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.scaler1 = Scaler(channels)
        self.scaler2 = Scaler(channels)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.PLKB(x)
        x1_scaler = self.scaler1(x - x1)

        x2 = self.DWConv_3(x)
        x2_scaler = self.scaler2(x - x2)

        x1 = x1 * x2_scaler
        x2 = x2 * x1_scaler

        out = self.act(self.conv1(torch.cat((x1, x2), dim=1)))
        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class RFMG(nn.Module):
    '''
    Residual Feature Modulation Group (RFMG)
    '''

    def __init__(self, channels, large_kernel, split_group):
        super(RFMG, self).__init__()
        #self.DyT1 = DyT(channels)
        #self.DyT2 = DyT(channels)
        self.HFDB = HFDB(channels, large_kernel, split_group)
        self.CGFN = CGFN(channels, large_kernel, split_group)
        self.norm1 = LayerNorm(channels, data_format='channels_first')
        self.norm2 = LayerNorm(channels, data_format='channels_first')

    def forward(self, x):
        x = self.HFDB(self.norm1(x)) + x
        x = self.CGFN(self.norm2(x)) + x
        return x


@ARCH_REGISTRY.register()
class LKMN(nn.Module):
    def __init__(self, in_channels, channels, out_channels, upscale, num_block, large_kernel, split_group):
        super(LKMN, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        # self.layers = Layers(channels, num_block, large_kernel=large_kernel, split_factor=split_factor)
        self.layers = nn.Sequential(*[RFMG(channels, large_kernel, split_group) for _ in range(num_block)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.upsampler = nn.Sequential(
            nn.Conv2d(channels, (upscale ** 2) * out_channels, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )
        self.act = nn.GELU()

    def forward(self, input):
        out_fea = self.conv_first(input)
        out = self.layers(out_fea)
        out = self.act(self.conv(out))
        output = self.upsampler(out + out_fea)
        return output



# from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
# net = LKMN(in_channels=3, channels=36, out_channels=3, upscale=4, num_block=8, large_kernel=31, split_group=4)  # 定义好的网络模型,实例化
# # print(net)
# input = torch.randn(1, 3, 320, 180)  # 1280*720---(640, 360)---(427, 240)---(320, 180)
# print(flop_count_table(FlopCountAnalysis(net, input)))