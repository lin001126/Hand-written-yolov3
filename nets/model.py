import torch
import torch.nn as nn
from collections import OrderedDict
from nets.darknet import darknet53

anchors = [[[116, 90], [156, 198], [373, 326]], [[30, 61], [62, 45], [59, 119]], [[10, 13], [16, 30], [33, 23]]]
classes = 1
pretrained = "./weights/darknet53_weights.pth"  # set empty to disable


class Model(nn.Module):
    def __init__(self, is_training=True):
        super(Model, self).__init__()
        self.training = is_training
        #  backbone
        self.backbone = darknet53(pretrained)
        _out_filters = self.backbone.layers_out_filters
        #  y1支路的剩余部分网络，即为6个DBL结构+1层conv输出
        final_out_filter0 = len(anchors[0]) * (5 + classes) # 表示y1支路生成了三个锚框，参数量为3*（80+4+1）
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  y2支路的网络特征融合部分
        final_out_filter1 = len(anchors[1]) * (5 + classes) # 表示y2支路生成了三个锚框，参数量为3*（80+4+1）
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest') # 上采样
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(anchors[2]) * (5 + classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        # 1     512+256，256， 1
        # 2     256，    512， 3
        # 3     512，    256， 1
        # 4     256，    512， 3
        # 5     512，    256， 1
        # 6     256，    512， 3

        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.embedding2, x2_in)
        return out0, out1, out2
