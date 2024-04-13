'''
Description: The file will be used split fracture images and normal images according AP or LA.
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-10 04:12:15
LastEditors: ShuaiLei
LastEditTime: 2024-04-10 04:23:49
'''
import os
import shutil


def split_image_according_AP_or_LA(images_folder, AP_folder, LA_folder):
    """
    The function will be used to split images to AP an d LA.
    param: images_folder: The images folder.
    param: AP_folder: The splited AP images save path.
    param: LA_folder: The splited LA images save path.
    """
    os.makedirs(AP_folder, exist_ok=True)
    os.makedirs(LA_folder, exist_ok=True)
    for file in os.listdir(images_folder):
        if "AP" in file:
            shutil.copy(os.path.join(images_folder, file), os.path.join(AP_folder, file))
        if "LA" in file:
            shutil.copy(os.path.join(images_folder, file), os.path.join(LA_folder, file))


if __name__ == "__main__":
    split_image_according_AP_or_LA("dataset/spine_fracture/drr/all/fracture_images",
                                   "dataset/spine_fracture/drr/AP/fracture_images",
                                   "dataset/spine_fracture/drr/LA/fracture_images")