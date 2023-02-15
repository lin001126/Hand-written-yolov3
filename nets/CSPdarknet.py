import torch
import torch.nn as nn

reluname = "frelu"
bottleneck = "se"


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


def autoPad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = CBL(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # 320, 320, 12 => 320, 320, 64
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], 1
            )
        )


class CBL(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autoPad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        if reluname == 'frelu':
            self.act = FReLU(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
            # print('1')
        else:
            self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SEBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.se=SE(c1,c2,ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        b, c, _, _ = x.size()
        y = self.avgpool(x1).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        out = x1 * y.expand_as(x1)

        # out=self.se(x1)*x1
        return x + out if self.add else out


class TransformerLayer(nn.Module):

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = CBL(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class CSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # CSPlayer
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c1, c_, 1, 1)
        self.cv3 = CBL(2 * c_, c2, 1)  # act=FReLU(c2)
        # 残差
        if bottleneck == 'se':
            # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
            self.m = nn.Sequential(*(SEBottleneck(c_, c_, shortcut) for _ in range(n)))
        elif bottleneck == 't':
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
            self.m = TransformerBlock(c_, c_, 4, n)
        else:
            self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
            , dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth,  pretrained):
        super().__init__()
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64

        # focus网络结构进行特征提取
        # 640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        self.focus = Focus(3, base_channels, k=3)

        # 320, 320, 64 -> 160, 160, 128
        # 完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # resblock1

        self.layer2 = nn.Sequential(
            # 320, 320, 64 -> 160, 160, 128
            CBL(base_channels, base_channels * 2, 3, 2),
            # 160, 160, 128 -> 160, 160, 128
            CSP(base_channels * 2, base_channels * 2, base_depth),
        )

        #   160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        #   这里出去一条yolo_head 80, 80, 256
        #    进行加强特征提取网络FPN的构建
        self.layer3 = nn.Sequential(
            CBL(base_channels * 2, base_channels * 4, 3, 2),
            CSP(base_channels * 4, base_channels * 4, base_depth * 3),
        )

        #   80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        #   这里出去一条yolo_head40, 40, 512
        #    进行加强特征提取网络FPN的构建
        # -----------------------------------------------#
        self.layer4 = nn.Sequential(
            CBL(base_channels * 4, base_channels * 8, 3, 2),
            CSP(base_channels * 8, base_channels * 8, base_depth * 3),
        )

        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # 这里是最后一条
        # -----------------------------------------------#
        self.layer5 = nn.Sequential(
            CBL(base_channels * 8, base_channels * 16, 3, 2),
            SPP(base_channels * 16, base_channels * 16),
            CSP(base_channels * 16, base_channels * 16, base_depth, shortcut=False),
        )
        if pretrained == True:
            checkpoint = torch.load('C:/Users/zjj/Desktop/result/150epoch_frelu/epoch_last.pt')
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights")

    def forward(self, x):
        x = self.focus(x)
        x = self.layer2(x)

        x = self.layer3(x)
        # 80, 80, 256
        out3 = x
        x = self.layer4(x)
        # 40, 40, 512
        out4 = x
        # 20, 20, 1024
        x = self.layer5(x)
        out5 = x
        return out3, out4, out5
