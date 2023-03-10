import torch


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.25, nms_thres=0.4):
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    box_corner[:, :, 0] = prediction[:, :, 0]
    box_corner[:, :, 1] = prediction[:, :, 1]
    box_corner[:, :, 2] = prediction[:, :, 2] + prediction[:, :, 2]
    box_corner[:, :, 3] = prediction[:, :, 3] + prediction[:, :, 3]
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # ?????????????????????????????????
        # conf_thres = torch.max(image_pred[:, 4])
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        # ?????????
        image_pred = image_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # ????????????????????????????????????
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # print('class {} {}'.format(class_conf, class_pred))
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # ?????????????????????
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # ???????????????
            detections_class = detections[detections[:, -1] == c]
            # ????????????????????????????????????
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # ??????????????????
            max_detections = []
            while detections_class.size(0):
                # ????????????????????????????????????????????????
                max_detections.append(detections_class[0].unsqueeze(0))
                # ???????????????????????????????????????
                if len(detections_class) == 1:
                    break
                # ???????????????????????????IoU
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # ?????? IoU >= NMS
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # ????????????????????????
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))
    return output
