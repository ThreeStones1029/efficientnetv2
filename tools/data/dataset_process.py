'''
Description: this file will be used split dataset.
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-19 05:08:26
LastEditors: ShuaiLei
LastEditTime: 2024-04-19 08:50:26
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math


def save_images_and_labels_to_txt(images_path, images_label, file_name):
    """
    The function is used to save split infomation into txt.
    param: images_path: The images path list.
    param: images_label: The images label list.
    param: file_name: The txt save path.
    """
    assert len(images_path) == len(images_label), "length of images_path and labels must equality, please check the images_path and labels."
    with open(file_name, 'w') as file:
        for image_path, label in zip(images_path, images_label):
            file.write(f'{image_path}, {label}\n')
    print(file_name, "save successfully!") 


def read_images_and_labels_from_txt(file_name):
    """
    The function is used to read split images_path and images_label infomation from txt.
    param: file_name: the images_path and images_label txt file.
    """
    images_path = []
    images_label = []
    with open(file_name, 'r') as file:
        for line in file:
            pair = line.strip().split(',')
            images_path.append(pair[0].strip())
            images_label.append(int(pair[1].strip()))
    return images_path, images_label


def read_split_dataset(root_folder, split_ratio = {"train": 0.6, "val": 0.2, "test": 0.2}, save_txt=True, plot_image=True):
    """
    The random split dataset.
    param: root_folder: the dataset.
    param: split_ratio: the split part and ratio.
    param: plot_image: draw class distribution.
    """
    np.random.seed(0)
    assert os.path.exists(root_folder), "dataset root: {} does not exist.".format(root_folder)
    # 遍历文件夹，一个文件夹对应一个类别
    class_name_list = [folder_name for folder_name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder_name))]
    class_name_list.sort()
    classname2classid = dict((k, v) for v, k in enumerate(class_name_list))
    classid2classname = json.dumps(dict((val, key) for key, val in classname2classid.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(classid2classname)
    exts = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    every_class_num = []
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    test_images_path = []
    test_images_label = []
    # 遍历每个文件夹下的文件
    for class_name in class_name_list:
        class_folder_path = os.path.join(root_folder, class_name)
        file_name_list = [file_name for file_name in os.listdir(class_folder_path) if os.path.splitext(file_name)[-1] in exts]
        np.random.shuffle(file_name_list)
        class_number = len(file_name_list)
        every_class_num.append(class_number)
        class_id = classname2classid[class_name]
        start = 0
        end = 0
        for i, (dataset_part_name, ratio) in enumerate(split_ratio.items()):
            start = end
            if i == len(split_ratio.keys()) - 1:
                part_file_name_list = file_name_list[start:]
            else:
                end = start + math.ceil(ratio * class_number)
                part_file_name_list = file_name_list[start: end]
            if dataset_part_name == "train":
                for file_name in part_file_name_list:
                    train_images_path.append(os.path.join(class_folder_path, file_name))
                    train_images_label.append(class_id)
            if dataset_part_name == "val":
                for file_name in part_file_name_list:
                    val_images_path.append(os.path.join(class_folder_path, file_name))
                    val_images_label.append(class_id)
            if dataset_part_name == "test":
                for file_name in part_file_name_list:
                    test_images_path.append(os.path.join(class_folder_path, file_name))
                    test_images_label.append(class_id)

    print("dataset split successfully!")
    print("total images number: {}".format(len(train_images_path) + len(val_images_path) + len(test_images_path)))
    print("{} images for train.".format(len(train_images_path)))
    print("{} images for val.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))
        
    assert len(train_images_path) > 0, "number of train images must greater than 0."
    assert len(val_images_path) > 0, "number of val images must greater than 0."

    if save_txt:
        save_images_and_labels_to_txt(train_images_path, train_images_label, os.path.join(root_folder, "train.txt"))
        save_images_and_labels_to_txt(val_images_path, val_images_label, os.path.join(root_folder, "val.txt"))
        save_images_and_labels_to_txt(test_images_path, test_images_label, os.path.join(root_folder, "test.txt"))
    
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(class_name_list)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(class_name_list)), class_name_list)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('class distribution')
        plt.savefig(os.path.join(root_folder, "class_distribution.png"))
    return train_images_path, train_images_label, val_images_path, val_images_label


if __name__ == "__main__":
    read_split_dataset("dataset/spine_fracture/cut_drr/all", split_ratio={"train": 0.6, "val": 0.2, "test": 0.2})