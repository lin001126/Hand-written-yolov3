import os
import cv2
import torch
from nets.model import Model
from nets.yoloModel import YoloModel
from nets.yolo_loss import YOLOLoss
from utils import non_max_suppression
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 1
w, h = (416, 416)
# w, h = (640, 640)
anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
# anchors = [[[9, 14], [12, 16], [10, 19]], [[14, 19], [13, 22], [17, 26]], [[22, 36], [32, 53], [55, 106]]]
classes = 1
model_path = r'models/epoch_last.pt'


def my_cal_iou_xyxy(box1,box2):
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

def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net = Model()
    net= YoloModel(1,pretrained=False)
    net.load_state_dict(torch.load(model_path))

    net = net.to(device).eval()
    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(anchors, classes, (w, h)))
    test_pic_path = './test_images/'
    for pic in os.listdir(test_pic_path):
        img = cv2.imread(os.path.join(test_pic_path, pic), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        image = img.copy()
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).to(device)
        image = torch.tensor(image)
        print(type(image))
        # inference
        with torch.no_grad():
            image = image.unsqueeze(0)
            outputs = net(image)

            print(outputs[0].shape)
            print(outputs[1].shape)
            print(outputs[2].shape)
            output_list = []

            label_path = r'E:\dataset\UA-DETRAC\lab1\MVI_40191__lab00921.txt'
            labels = np.loadtxt(label_path)
            labels =labels[np.newaxis, :, :]
            print((torch.from_numpy(labels)))
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i], torch.from_numpy(labels)))
            # output = torch.cat(output_list, 1)
            batch_b = []
            for output in output_list:
                batch_b.append(output[-1])
            batch_detections = batch_b
            # batch_detections = non_max_suppression(output, classes, conf_thres=0.5, nms_thres=0.45)
            # print(batch_detections)
        label_path = r'E:\dataset\UA-DETRAC\lab1\MVI_40191__lab00921.txt'
        labels = np.loadtxt(label_path)
        batch_detection = []
        x=640
        for cla, x1, y1, x2, y2 in labels:
            box1 = [x1*x, y1*x, x2*x, y2*x]
            max_iou = 0
            mtemp = batch_detections[0][0]
            for tmp in batch_detections[0]:
                iou = my_cal_iou_xyxy(box1, tmp)
                if iou > max_iou:
                    max_iou = iou
                    mtemp = tmp
            batch_detection.append(mtemp)

        with open("./coco.names", "r") as fi:
            classes_name = fi.read().split("\n")[:-1]
            for idx, detections in enumerate(batch_detection):
                print(detections)
                if detections is not None:

                    x1 = detections[0]
                    y1 = detections[1]
                    x2 = detections[2]
                    y2 = detections[3]

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow('pic', img)
            cv2.waitKey(0)


if __name__ == "__main__":
    test()
