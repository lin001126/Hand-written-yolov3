import os
import numpy as np
import cv2
import torchvision

import data_transforms
from torch.utils.data import Dataset
path = r'E:/dataset/UA-DETRAC/'


class COCODataset(Dataset):
    def __init__(self, list_path, img_size, is_training=False, is_debug=False):
        self.img_files = []
        self.label_files = []
        iter1 = 0
        for line in open(list_path, 'r', encoding='utf-8'):
            image_path, labels = line.split(" ", 1)
            a, b = image_path.split("/", 1)
            c, d = b.split(".", 1)
            if(c == "img00001" ):
            # if(c == "img00001" or c == "img00101" or c == "img00201" or c == "img00301" or c == "img00401" or c == "img00501"):
                image_path = image_path.replace("/", "__")

                label_path = image_path.replace('img', 'lab').replace('.jpg', '.txt')
                if os.path.isfile(path + 'lab1/' + label_path):
                    self.img_files.append(path + 'picture_test/' + image_path)
                    self.label_files.append(path + 'lab1/' + label_path)
            else:
                pass
        self.img_size = img_size  # (w, h)
        self.max_objects = 1
        self.is_debug = is_debug
        print("len  {} {} ".format(len(self.img_files), len(self.label_files)))

        #  transforms and augmentation
        self.transforms = data_transforms.Compose()
        if is_training:
            self.transforms.add(data_transforms.ImageBaseAug())
        # self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(data_transforms.ResizeImage(self.img_size))
        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)

        if img is None:
            raise Exception("Read image error: {}".format(img_path))
        ori_h, ori_w = img.shape[:2]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # print(True if int(max(labels[:,:1]))>80 else False)
        else:
            labels = np.zeros((1, 5), np.float32)

        sample = {'image': img, 'label': labels}
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample["image_path"] = img_path
        sample["origin_size"] = str([ori_w, ori_h])
        return sample

    def __len__(self):
        return len(self.img_files)
