import os

import cv2
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from coco_dataset import COCODataset
from nets.model import Model
from nets.yoloModel import YoloModel
from nets.yolo_loss import YOLOLoss

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
batch_size = 1
# w, h = (416, 416)
w, h = (640, 640)
path = r'E:/dataset/UA-DETRAC/'
train_path = path + 'train_gt.txt'
label_path = r'E:/dataset/UA-DETRAC/lab1/MVI_20011__lab00001.txt'
pic_path = r'E:/dataset/UA-DETRAC/picture_test/MVI_20011__img00001.jpg'
# anchors = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]]
anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
# anchors = [[[9, 14], [12, 16], [10, 19]], [[14, 19], [13, 22], [17, 26]], [[22, 36], [32, 53], [55, 106]]]
classes = 1
save_model_path = './models'
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

model_path=r'C:\Users\zjj\Desktop\result\60pic_100epoch_noadd\epoch_last.pt'

def draw(x_axis, y1_axis, y2_axis):
    plt.figure('loss image')
    plt.xlabel('iter')
    plt.ylabel('loss/iou')
    plt.plot(x_axis, y1_axis)
    plt.plot(x_axis, y2_axis)
    plt.legend(['loss', 'avg_iou'])
    plt.savefig('loss.jpg')


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():
    # net = Model()
    net = YoloModel(classes=1)
    # net.load_state_dict(torch.load(model_path))
    base_lr = 1e-3
    # optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=4e-05)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-05)
    net = net.to(device)

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(anchors, classes, (w, h)))
    # DataLoader
    dataloader = torch.utils.data.DataLoader(COCODataset(train_path, (w, h), is_training=False),
                                             batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    max_epoch = 150
    net.train()
    data_total_groups = len(dataloader) // batch_size
    x_axis = []
    y1_axis = []
    y2_axis = []
    anch_ious_list = []

    for epoch in range(max_epoch):
        print('epoch-----------------{}'.format(epoch))

        set_lr(optimizer, base_lr * 0.9)  # 设置阶梯下降学习率

        for iter, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            images = images.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls", "anch_ious_list"]
            losses = []
            avg_iou = []
            avg_loss = []

            losstest = []
            losstest1 = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_items = yolo_losses[i](outputs[i], labels)
                _loss_item = _loss_items[:-3]  # "total_loss", "x", "y", "w", "h", "conf", "cls"
                losstest.append(_loss_items[-2])
                losstest1.append(_loss_items[-1])
                if (len(_loss_items[-3]) == 0):
                    pass
                else:
                    anch_ious_list.append(sum(_loss_items[-3]) / len(_loss_items[-3]))  # "anch_ious_list"
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            anch_ious = sum(anch_ious_list) / len(anch_ious_list)
            avg_iou.append(anch_ious)

            loss = losses[0]
            avg_loss.append(loss)
            loss.backward()
            optimizer.step()

            if iter > 0 and iter % 100 == 0:
                _loss = loss.item()
                print("epoch[%d|%d] iter[%d|%d] loss=%.4f avg_iou=%.4f" % (
                    epoch, max_epoch, iter, len(dataloader), _loss, anch_ious))
                x_axis.append(iter + data_total_groups * epoch)
                y1_axis.append(_loss)
                y2_axis.append(anch_ious)
                # draw(x_axis, y1_axis, y2_axis)
        if epoch % 10 == 0:
            save_checkpoint(net.state_dict(), epoch)
        # 验证集，能跑就行，别细看代码。
        ap = getAP(losstest1, label_path)
        print('val: epoch: {}/{} iou: {} loss: {} AP: {}'.format(epoch, max_epoch,
                                                                 (sum(avg_iou) / len(avg_iou)),
                                                                 (sum(avg_loss) / len(avg_loss)), ap))
        fi = open('info.txt', 'a')
        print('{} {} {} {} {}'.format(epoch, max_epoch, (sum(avg_iou) / len(avg_iou)), (sum(avg_loss) / len(avg_loss)),
                                      ap), file=fi)
        fi.close()

        _loss = loss.item()
    mytest(outputs, yolo_losses, losstest, losstest1, label_path, pic_path)
    save_checkpoint(net.state_dict(), "last")


def my_cal_iou_xyxy(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # 计算iou
    iou = intersection / union
    return iou


def getAP(losstest1, label_path):
    # label_path = r'D:/BaiduNetdiskDownload/UA-DETRAC/lab1/MVI_63563__lab00001.txt'
    labels = np.loadtxt(label_path).reshape(-1, 5)
    temp = []

    for losss in losstest1:
        for lossss in losss:
            xx1 = lossss[0]
            yy1 = lossss[1]
            xx2 = lossss[2]
            yy2 = lossss[3]
            temp.append([xx1, yy1, xx2, yy2])
    num = 0
    ssum = 0
    for cla, x1, y1, x2, y2 in labels:
        num += 1
        box1 = [x1 * 416, y1 * 416, x2 * 416, y2 * 416]
        max_iou = 0
        for tmp in temp:
            iou = my_cal_iou_xyxy(box1, tmp)
            if iou > max_iou:
                max_iou = iou
        k = (int)(max_iou * 10) + 1
        ssum += k
    # print('{} {} {}'.format(ssum, num, (float)(ssum/num/11)))
    return (float)(ssum / num / 11)


def mytest(outputs, yolo_losses, losstest, losstest1, labei_path, pic_path):
    # pic_path = r'D:/BaiduNetdiskDownload/UA-DETRAC/picture_test/MVI_20011__img00001.jpg'
    img = cv2.imread(os.path.join(pic_path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
    image = img.copy()
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).to(device)
    image = torch.tensor(image)

    # label_path = r'D:/BaiduNetdiskDownload/UA-DETRAC/lab1/MVI_63563__lab00001.txt'
    labels = np.loadtxt(label_path)
    # for cla, x0, y0, x1, y1 in labels:
    #     cv2.rectangle(img, (int(x0 * 416), int(y0 * 416)), (int(x1 * 416), int(y1 * 416)), (0, 0, 0), 1, cv2.LINE_AA)
    #
    # for losss in losstest:
    #     for lossss in losss:
    #         x0 = int(lossss[0])
    #         y0 = int(lossss[1])
    #         x1 = int(lossss[2]) - x0
    #         y1 = int(lossss[3]) - y0
    #         cv2.rectangle(img, (x0, y0, x1, y1), (0, 255, 0), 1, cv2.LINE_AA)

    for losss in losstest1:
        for lossss in losss:
            x0 = int(lossss[0])
            y0 = int(lossss[1])
            x1 = int(lossss[2]) - x0
            y1 = int(lossss[3]) - y0
            cv2.rectangle(img, (x0, y0, x1, y1), (255, 0, 0), 1, cv2.LINE_AA)

    # cv2.imshow('pic', img)
    # cv2.waitKey(0)


def save_checkpoint(state_dict, param):
    checkpoint_path = os.path.join(save_model_path, "epoch_%s.pt" % param)
    torch.save(state_dict, checkpoint_path)


if __name__ == "__main__":
    train()
