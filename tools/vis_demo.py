'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-05-04 10:12:56
LastEditors: ShuaiLei
LastEditTime: 2024-05-04 10:26:33
'''
import root_path
from tools.vis.bbox_gt_visualize import VisCoCo


if __name__ == "__main__":
    VisCoCo("dataset/spine_fracture/xray/annotations/fracture_normal.json",
            "dataset/spine_fracture/xray/images",
            "dataset/spine_fracture/xray/gt").visualize_bboxes_in_images()