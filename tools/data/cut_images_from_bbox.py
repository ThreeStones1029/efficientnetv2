'''
Description:                               
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-05-04 10:28:05
LastEditors: ShuaiLei
LastEditTime: 2024-07-14 02:40:42
'''
import os
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from tools.bbox.bbox_process import get_cut_bbox
from tools.io.common import create_folder


def get_cut_images_from_gt_bboxes(images_folder, coco_json_file, cut_images_folder):
    """
    This function will be used to cut images according bbox.
    param: images_folder: the images folder.
    param: coco_json_file: the coco format json file.
    param: cut_images_folder: the cut images save folder.
    """
    gt = COCO(coco_json_file)
    fracture_images_folder = create_folder(os.path.join(cut_images_folder, "fracture_images"))
    normal_images_folder = create_folder(os.path.join(cut_images_folder, "normal_images"))
    for image_info in tqdm(gt.dataset["images"], desc="cutting images according gt bboxes"):
        file_name = image_info["file_name"]
        image = Image.open(os.path.join(images_folder, file_name)).convert('RGB')
        width, height = image.size
        i = 0
        for ann in gt.imgToAnns[image_info["id"]]:
            cut_bbox = get_cut_bbox(ann["bbox"], width, height, expand_coefficient=1.5)
            cut_image = image.crop((cut_bbox[0], cut_bbox[1], cut_bbox[2], cut_bbox[3]))
            i += 1
            if ann["category_name"] == "fracture":
                cut_image.save(os.path.join(fracture_images_folder, file_name.split(".")[0] + "_" + str(i) + ".png"))
            if ann["category_name"] == "normal":
                cut_image.save(os.path.join(normal_images_folder, file_name.split(".")[0] + "_" + str(i) + ".png"))