'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-05-04 09:56:41
LastEditors: ShuaiLei
LastEditTime: 2024-05-04 10:13:41
'''
from tools.io.common import load_json_file, save_json_file


def rename_images_in_coco_json_file(coco_json_file):
    """
    This function will be used to rename json file.
    param: coco_json_file: the coco json file from label_studio.
    """
    data = load_json_file(coco_json_file)
    for image in data["images"]:
        image["file_name"] = image["file_name"].split("-")[1]
    save_json_file(data, coco_json_file)