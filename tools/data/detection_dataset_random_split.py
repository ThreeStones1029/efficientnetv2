'''
Description: this file will be used to split detection dataset to trainval and test dataset
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-05-04 13:21:53
LastEditors: ShuaiLei
LastEditTime: 2024-05-04 13:23:57
'''
import os
import numpy as np
import shutil
from tools.io.common import load_json_file, save_json_file
    
    
def random_split_coco_dataset(images_folder_path, annotation_file, output_folder_path, split_info_dict):
    """
    随机划分json文件,并划分好相应的数据集
    """
    os.makedirs(output_folder_path, exist_ok=True)
    # 读取annotations.json文件
    dataset = load_json_file(annotation_file)
    # 提取images, annotations, categories
    # 随机打乱数据
    np.random.shuffle(dataset["images"])
    start_index = 0
    end_index = 0
    def filter_annotations(annotations, image_ids):
        return [ann for ann in annotations if ann["image_id"] in image_ids]
    for split_part_name, ratio in split_info_dict.items():
        end_index += int(ratio * len(dataset["images"]))
        split_part_images = dataset["images"][start_index:end_index]
        start_index += int(ratio * len(dataset["images"]))
        split_part_folder = os.path.join(output_folder_path, split_part_name)
        os.makedirs(split_part_folder, exist_ok=True)
        for img in split_part_images:
            shutil.copy(os.path.join(images_folder_path, img["file_name"]), os.path.join(split_part_folder, img["file_name"]))
        split_part_annotations = filter_annotations(dataset["annotations"], [img["id"] for img in split_part_images])
        split_part_data = {"info": dataset["info"], "images": split_part_images, "annotations": split_part_annotations, "categories": dataset["categories"]}
        save_json_file(split_part_data, os.path.join(output_folder_path,"bbox_" + split_part_name + ".json"))
    print("数据集划分完成！")