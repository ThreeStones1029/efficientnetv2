'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-03-31 04:04:02
LastEditors: ShuaiLei
LastEditTime: 2024-04-13 08:50:56
'''
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    

class ValDataSet(Dataset):
    """测试数据集"""
    def __init__(self, images_path: list, transform=None):
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        if self.transform is not None:
            img = self.transform(img)
        return img

    @staticmethod
    def collate_fn(batch):
        images = batch
        images = torch.stack(images, dim=0)
        return images
    

class ValDataSetNoRead(Dataset):
    """测试数据集 不读取"""
    def __init__(self, images: list, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        if self.transform is not None:
            img = self.transform(img)
        return img

    @staticmethod
    def collate_fn(batch):
        images = batch
        images = torch.stack(images, dim=0)
        return images
