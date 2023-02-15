import torch
import torch.nn as nn
import numpy as np
import math
from utils import bbox_iou
from utils import non_max_suppression

print(torch.cuda.is_available())
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.label_smoothing = 0
        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.anchor_new = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]]
        input_shape = [416, 416]
        self.threshold = 4
        self.anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 1)
        self.cuda = True

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def box_giou(self, b1, b2):
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh/2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh/2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return iou, giou

    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def get_pred_boxes(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w):
        bs = len(targets)
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        return pred_boxes

    def forward(self, input, targets=None):  # input表示y1,y2,y3的前向传播结果，targets表示对应的GroundTruth标签
        bs = input.size(0)
        l = 0
        in_h = input.size(2)  # 表示y方向的格子数，例如13，26，52
        in_w = input.size(3)  # 表示x方向的格子数，例如13，26，52
        if in_h == 26:
            l = 1
        elif in_h == 52:
            l = 2
        stride_h = self.img_size[1] / in_h  # 表示y方向每个格子所占像素px
        stride_w = self.img_size[0] / in_w  # 表示x方向每个格子所占像素px
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        scaled_anchors_new = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchor_new[l]]
        prediction = input.view(bs, len(self.anchors_mask[l]),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = torch.sigmoid(prediction[..., 2])  # Width
        h = torch.sigmoid(prediction[..., 3])  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        if targets is not None:
            y_true, noobj_mask = self.get_target(l, targets, scaled_anchors, in_h, in_w)
            pred_boxes = self.get_pred_boxes(l, x, y, w, h, targets, scaled_anchors, in_h, in_w)

            if self.cuda:
                y_true = y_true.type_as(x)
            loss = 0
            n = torch.sum(y_true[..., 4] == 1)
            if n != 0:

                iou, giou = self.box_giou(pred_boxes, y_true[..., :4])
                loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])
                loss_cls = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))
                loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

                tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))
            else:
                tobj = torch.zeros_like(y_true[..., 4])
                iou = torch.zeros_like(y_true[..., 4])
                giou = torch.zeros_like(y_true[..., 4])
            loss_conf = torch.mean(self.BCELoss(conf, tobj))
            loss += loss_conf * self.balance[l] * self.obj_ratio

            return loss, iou[y_true[..., 4] == 1], y_true[y_true[..., 4] == 1] * stride_h, pred_boxes[y_true[..., 4] == 1] * stride_h
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor


            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)

            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
                int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
            grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
                int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

            scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
            anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
            anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)

            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

            # pred_boxes[..., 0] = x.data + grid_x
            # pred_boxes[..., 1] = y.data + grid_y
            # pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            # pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h


            t1 = x.data + grid_x
            t2 = y.data + grid_y
            t3 = torch.exp(w.data) * anchor_w
            t4 = torch.exp(h.data) * anchor_h
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def cal_iou_xyxy(box1,box2):
        x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
        x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
        #计算两个框的面积
        s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
        s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

        #计算相交部分的坐标
        xmin = max(x1min,x2min)
        ymin = max(y1min,y2min)
        xmax = min(x1max,x2max)
        ymax = min(y1max,y2max)

        inter_h = max(ymax - ymin + 1, 0)
        inter_w = max(xmax - xmin + 1, 0)

        intersection = inter_h * inter_w
        union = s1 + s2 - intersection

        #计算iou
        iou = intersection / union
        return iou


    def get_target(self, l, targets,  anchors, in_w, in_h):
        bs = len(targets)
        no_obj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        box_best_ratio = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            batch_target[:, [0, 2]] = targets[b][:, [1, 3]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [2, 4]] * in_h
            batch_target[:, 4] = targets[b][:, 0]
            batch_target = batch_target.cpu()
            ratios_of_gt_anchors = torch.unsqueeze(batch_target[:, 2:4], 1) / torch.unsqueeze(torch.FloatTensor(anchors), 0)
            ratios_of_anchors_gt = torch.unsqueeze(torch.FloatTensor(anchors), 0) / torch.unsqueeze(batch_target[:, 2:4], 1)
            ratios = torch.cat([ratios_of_anchors_gt, ratios_of_gt_anchors], dim=-1)
            max_ratios, _ = torch.max(ratios, dim=-1)
            for t, ratio in enumerate(max_ratios):
                over_threshold = ratio < self.threshold
                over_threshold[torch.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue

                    i = torch.floor(batch_target[t, 0]).long()
                    j = torch.floor(batch_target[t, 1]).long()

                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[b, k, local_j, local_i] != 0:
                            if box_best_ratio[b, k, local_j, local_i] > ratio[mask]:
                                y_true[b, k, local_j, local_i, :] = 0
                            else:
                                continue

                        c = batch_target[t, 4].long()

                        no_obj_mask[b, k, local_j, local_i] = 0

                        y_true[b, k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[b, k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[b, k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[b, k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[b, k, local_j, local_i, 4] = 1
                        y_true[b, k, local_j, local_i, c + 5] = 1
                        box_best_ratio[b, k, local_j, local_i] = ratio[mask]
        return y_true, no_obj_mask
