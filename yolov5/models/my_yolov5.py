'''
Description: 重写yolov5网络结构
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-06-03 01:29:48
LastEditors: ShuaiLei
LastEditTime: 2024-06-04 13:09:36
'''
import argparse
import torch.nn as nn
import torch
import warnings
from pathlib import Path
import sys
import platform
import os 
import math
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.general import print_args
from utils.torch_utils import profile,select_device, fuse_conv_and_bn, scale_img

depth_multiple_dict = {"yolov5s": 0.33,"yolov5m": 0.67, "yolov5l": 1.0, "yolov5x": 1.33}
width_multiple_dict = {"yolov5s": 0.5, "yolov5m": 0.75, "yolov5l": 1.0, "yolov5x": 1.25}
p3_anchors = [[10, 13], [16, 30], [33, 23]]
p4_anchors = [[30, 61], [62, 45], [59, 119]]
p5_anchors = [[116, 90], [156, 198], [373, 326]]


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation x*sigmoid(x)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_hidden = int(c1 * e)
        self.conv1 = Conv(c1, c_hidden, k=1, s=1)
        self.conv2 = Conv(c_hidden, c2, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_hidden = int(c1 * e)
        self.conv1 = Conv(c1, c_hidden, k=1, s=1)
        self.conv2 = Conv(c1, c_hidden, k=1, s=1)
        self.conv3 = Conv(2 * c_hidden, c2, k=1, s=1)
        self.m = nn.Sequential(*(Bottleneck(c_hidden, c_hidden, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))
    

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_hidden = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, c_hidden, k=1, s=1)
        self.conv2 = Conv(c_hidden*4, c2, k=1, s=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.conv2(torch.cat((x, y1, y2, self.m(y2)), 1))
    

class Backbone(nn.Module):
    def __init__(self, c1, c2, width_multiple, depth_multiple):
        super().__init__()
        conv1_c_out = int(64 * width_multiple) # 第一个CBS的输出通道数目
        self.conv1 = Conv(c1, conv1_c_out, k=6, s=2, p=2)
        conv2_c_out = int(128 * width_multiple) # 第二个CBS的输出通道数目
        self.conv2 = Conv(conv1_c_out, conv2_c_out, k=3, s=2, p=1)
        self.c3_1 = C3(conv2_c_out, conv2_c_out, n=round(3 *depth_multiple), shortcut=True)
        conv3_c_out = int(256 *  width_multiple) # 第三个CBS的输出通道数目
        self.conv3 = Conv(conv2_c_out, conv3_c_out, k=3, s=2, p=1)
        self.c3_2 = C3(conv3_c_out, conv3_c_out, n=round(6*depth_multiple), shortcut=True)
        conv4_c_out = int(512 * width_multiple)
        self.conv4 = Conv(conv3_c_out, conv4_c_out, k=3, s=2, p=1)
        self.c3_3 = C3(conv4_c_out, conv4_c_out, n=round(9*depth_multiple), shortcut=True)
        conv5_c_out = int(1024 * width_multiple)
        self.conv5 = Conv(conv4_c_out, conv5_c_out, k=3, s=2, p=1)
        self.c3_4 = C3(conv5_c_out, conv5_c_out, n=round(3*depth_multiple), shortcut=True)
        self.sppf = SPPF(conv5_c_out, c2)

    def forward(self, x):
        conv1_y = self.conv1(x)
        conv2_y = self.conv2(conv1_y)
        c3_1_y = self.c3_1(conv2_y)
        conv3_y = self.conv3(c3_1_y)
        c3_2_y = self.c3_2(conv3_y)
        conv4_y = self.conv4(c3_2_y)
        c3_3_y = self.c3_3(conv4_y)
        conv5_y = self.conv5(c3_3_y)
        c3_4_y = self.c3_4(conv5_y)
        y = self.sppf(c3_4_y)
        return c3_2_y, c3_3_y, y
    

class FPN_PAN(nn.Module):
    def __init__(self, c1, depth_multiple, g=1, e=0.5):
        super().__init__()
        FPN_conv1_c_out = int(c1 * e) # 表示FPN模块的第一个卷积块的隐藏层
        self.FPN_conv1 = Conv(c1, FPN_conv1_c_out, k=1, s=1) # 表示FPN模块的第一个卷积块
        self.FPN_unsample1 = nn.Upsample(None, 2, 'nearest') # FPN第一个上采样
        self.FPN_c3_1 = C3(c1, FPN_conv1_c_out, n=round(3 * depth_multiple), shortcut=False, g=g, e=e) # FPN模块的第一个C3模块
        FPN_conv2_c_out = int(FPN_conv1_c_out * e) # 表示FPN模块的第二个卷积块的隐藏层
        self.FPN_conv2 = Conv(FPN_conv1_c_out, FPN_conv2_c_out, k=1, s=1) # 表示FPN模块的第二个卷积块
        self.FPN_unsample2 = nn.Upsample(None, 2, 'nearest') # FPN模块的第二个C3模块
        self.FPN_c3_2 = C3(FPN_conv1_c_out, FPN_conv2_c_out, n=round(3 * depth_multiple), shortcut=False, g=g, e=e) # FPN模块的第二个C3模块
        PAN_conv1_c_in = PAN_conv1_c_out = FPN_conv2_c_out
        self.PAN_conv1 = Conv(PAN_conv1_c_in, PAN_conv1_c_out, k=3, s=2, p=1) # PAN模块的第一个卷积块
        self.PAN_c3_1 = C3(PAN_conv1_c_out * 2, PAN_conv1_c_out * 2, n=round(depth_multiple * 3), shortcut=False, g=g, e=e) # PAN模块的第一个C3模块
        PAN_conv2_c_in = PAN_conv2_c_out = PAN_conv1_c_out * 2
        self.PAN_conv2 = Conv(PAN_conv2_c_in, PAN_conv2_c_out, k=3, s=2, p=1) # PAN模块的第二个卷积块
        self.PAN_c3_2 = C3(PAN_conv2_c_out * 2, PAN_conv2_c_out * 2, n=round(depth_multiple * 3), shortcut=False, g=g, e=e) # PAN模块的第二个C3模块


class YOLOv5(nn.Module):
    def __init__(self, c1, num_classes, width_multiple, depth_multiple):
        super().__init__()
        self.backbone = Backbone(c1, int(1024*width_multiple), width_multiple, depth_multiple)
        self.fpn_pan = FPN_PAN(int(1024*width_multiple), depth_multiple)
        self.head1 = nn.Conv2d(int(256 * width_multiple), 3*(num_classes + 5), 1, 1, 1, groups=1, dilation=1, bias=False)
        self.head2 = nn.Conv2d(int(512 * width_multiple), 3*(num_classes + 5), 1, 1, 1, groups=1, dilation=1, bias=False)
        self.head3 = nn.Conv2d(int(1024 * width_multiple), 3*(num_classes + 5), 1, 1, 1, groups=1, dilation=1, bias=False)

    def forward(self, x):
        backbone_c3_2_y, backbone_c3_3_y, backbone_y = self.backbone(x)
        fpn_conv1_y = self.fpn_pan.FPN_conv1(backbone_y)
        fpn_unsample1_y = self.fpn_pan.FPN_unsample1(fpn_conv1_y)
        fpn_c3_1_y = self.fpn_pan.FPN_c3_1(torch.cat((backbone_c3_3_y, fpn_unsample1_y), 1))
        fpn_conv2_y = self.fpn_pan.FPN_conv2(fpn_c3_1_y)
        fpn_unsample2_y = self.fpn_pan.FPN_unsample2(fpn_conv2_y)
        fpn_c3_2_y = self.fpn_pan.FPN_c3_2(torch.cat((backbone_c3_2_y, fpn_unsample2_y), 1))
        head1_y = self.head1(fpn_c3_2_y)
        pan_conv1_y = self.fpn_pan.PAN_conv1(fpn_c3_2_y)
        pan_c3_1_y = self.fpn_pan.PAN_c3_1(torch.cat((fpn_conv2_y, pan_conv1_y), 1))
        head2_y = self.head2(pan_c3_1_y)
        pan_conv2_y = self.fpn_pan.PAN_conv2(pan_c3_1_y)
        pan_c3_2_y = self.fpn_pan.PAN_c3_2(torch.cat((fpn_conv1_y, pan_conv2_y), 1))
        head3_y = self.head3(pan_c3_2_y)
        return head1_y, head2_y, head3_y
    
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        """
        # 将推理结果恢复到原图图片尺寸(逆操作)
        """
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    opt = parser.parse_args()
    print_args(vars(opt))
    device = select_device(opt.device)
    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    Model = YOLOv5
    model = Model(3, 4, width_multiple_dict["yolov5x"], depth_multiple_dict["yolov5x"]).to(device)
    model.forward(im)
    print(model)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)
    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)
    else:  # report fused model summary
        model.fuse()