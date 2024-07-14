'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-05-04 10:13:06
LastEditors: ShuaiLei
LastEditTime: 2024-07-14 13:49:09
'''
import root_path
from tools.data.label_studio import rename_images_in_coco_json_file
from tools.data.cut_images_from_bbox import get_cut_images_from_gt_bboxes
from tools.data.detection_dataset_random_split import random_split_coco_dataset



if __name__ == "__main__":
    # reanme json file from label_studio
    # rename_images_in_coco_json_file("dataset/spine_fracture/xray/annotations/result.json")

    # random split detection to train_val and test. the train_val will be used to train classify model and test will be used to test classify model.
    # random_split_coco_dataset(images_folder_path="dataset/spine_fracture/xray/images",
    #                           annotation_file="dataset/spine_fracture/xray/annotations/fracture_normal.json",
    #                           output_folder_path="dataset/spine_fracture/xray",
    #                           split_info_dict={"train_val": 0.8, "test":0.2})
    # cut train_val images for training classify model
    get_cut_images_from_gt_bboxes("/home/RT-DETR/rtdetr_paddle/datasets/TD20240705_LA/split_dataset/val",
                                  "/home/RT-DETR/rtdetr_paddle/datasets/TD20240705_LA/split_dataset/annotations/fracture_bbox_val.json",
                                  "/home/RT-DETR/rtdetr_paddle/datasets/TD20240705_LA/cut_dataset/val")