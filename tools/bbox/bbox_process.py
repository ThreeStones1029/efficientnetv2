'''
Description: this file will used to bounding box process.
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-18 11:42:26
LastEditors: ShuaiLei
LastEditTime: 2024-04-18 11:46:31
'''
def get_cut_bbox(bbox, width, height, expand_coefficient):
    # 根据bbox计算最大最小坐标
    x, y, w, h = bbox
    center_x, center_y = x + w/2, y + h/2
    expand_w = expand_coefficient * w
    expand_h = expand_coefficient * h
    new_min_x = center_x - expand_w / 2 if center_x - expand_w / 2 > 0 else 0
    new_min_y = center_y - expand_h / 2 if center_y - expand_h / 2 > 0 else 0
    new_max_x = center_x + expand_w / 2 if center_x + expand_w / 2 < width else width
    new_max_y = center_y + expand_h / 2 if center_y + expand_h / 2 < height else height
    return [int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)]