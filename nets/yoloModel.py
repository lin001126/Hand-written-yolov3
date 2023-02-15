import torch
import torch.nn as nn

from nets.CSPdarknet import CSP, CBL, CSPDarknet

anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloModel(nn.Module):
    def __init__(self, classes,  pretrained=True):
        super(YoloModel, self).__init__()


        # base_channels = int(wid_mul * 64)  # 64
        base_channels=32
        print(base_channels)
        # base_depth = max(round(dep_mul * 3), 1)  # 3
        # print(base_depth)
        base_depth=1
        #  backbone
        # 640, 640, 3
        self.backbone = CSPDarknet(base_channels, base_depth, pretrained)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p5_cbl = CBL(base_channels * 16, base_channels * 8, 1, 1)
        self.p5_upsampling = CSP(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.p4_cbl = CBL(base_channels * 8, base_channels * 4, 1, 1)
        self.p4_upsampling = CSP(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.p3p4_cbl = CBL(base_channels * 4, base_channels * 4, 3, 2)
        self.p3p4_csp = CSP(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.p4p5_cbl = CBL(base_channels * 8, base_channels * 8, 3, 2)
        self.p4p5_csp = CSP(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + classes) => 80, 80, 3 * (4 + 1 + classes)
        print(len(anchors_mask[2]))
        self.yolo_head1 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + classes) => 40, 40, 3 * (4 + 1 + classes)
        self.yolo_head2 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + classes) => 20, 20, 3 * (4 + 1 + classes)
        self.yolo_head3 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + classes), 1)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # neck:
        # [[-1, 1, Conv, [512, 1, 1]],
        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.p5_cbl(x0)

        #  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)

        #  [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_upsample, x1], 1)

        #  [-1, 3, C3, [512, False]],  # 13
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.p5_upsampling(P4)

        #  [-1, 1, Conv, [256, 1, 1]],
        # 40, 40, 512 -> 40, 40, 256
        P4 = self.p4_cbl(P4)

        #  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)

        #  [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, x2], 1)

        #  [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.p4_upsampling(P3)

        #  [-1, 1, Conv, [256, 3, 2]],
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.p3p4_cbl(P3)

        #  [[-1, 14], 1, Concat, [1]],  # cat head P4
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)

        #  [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.p3p4_csp(P4)

        #  [-1, 1, Conv, [512, 3, 2]],
        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.p4p5_cbl(P4)

        #  [[-1, 10], 1, Concat, [1]],  # cat head P5
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)

        #  [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.p4p5_csp(P5)

        #  yolo branch (batch_size,75,80,80)
        out2 = self.yolo_head1(P3)

        #  yolo branch (batch_size,75,40,40)
        out1 = self.yolo_head2(P4)

        #   yolo branch (batch_size,75,20,20)
        out0 = self.yolo_head3(P5)
        return out0, out1, out2
