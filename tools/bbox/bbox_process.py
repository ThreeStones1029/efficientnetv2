'''
Description: this file will used to bounding box process.
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-18 11:42:26
LastEditors: ShuaiLei
LastEditTime: 2024-04-19 04:41:53
'''
from tools.io.common import load_json_file, save_json_file


def get_cut_bbox(bbox, width, height, expand_coefficient):
    """
    according bbox compute min max x and y to cut image.
    param: bbox: the detection bbox
    param: width: the image width.
    param: height: the image height.
    param: expand_coefficient: the expand coefficient.
    """
    x, y, w, h = bbox
    center_x, center_y = x + w/2, y + h/2
    expand_w = expand_coefficient * w
    expand_h = expand_coefficient * h
    new_min_x = center_x - expand_w / 2 if center_x - expand_w / 2 > 0 else 0
    new_min_y = center_y - expand_h / 2 if center_y - expand_h / 2 > 0 else 0
    new_max_x = center_x + expand_w / 2 if center_x + expand_w / 2 < width else width
    new_max_y = center_y + expand_h / 2 if center_y + expand_h / 2 < height else height
    return [int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)]


def filter_low_score_bboxes(bbox_json_file, threshold):
    """
    filter low score bboxes for fracture classify.
    param: bbox_json_file: the detection result.
    param: threshold: the lowest score.
    """
    filter_bboxes = []
    bboxes = load_json_file(bbox_json_file)
    if "id" not in bboxes[0].keys():
        id = -1
        for ann in bboxes:
            if ann["score"] >= threshold:
                id += 1
                ann["id"] = id
                filter_bboxes.append(ann)
    else:
        for ann in bboxes:
            if ann["score"] >= threshold:
                filter_bboxes.append(ann)

    save_json_file(filter_bboxes, bbox_json_file)
