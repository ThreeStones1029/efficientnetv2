'''
Description: this function will be used to visualize fracture information and bbox
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-18 13:35:20
LastEditors: ShuaiLei
LastEditTime: 2024-05-06 11:49:06
'''
import numpy as np
from PIL import ImageDraw, ImageFont


def draw_bbox(image, bboxes, fontsize=20):
    """
    Draw bbox on image
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('arial.ttf', fontsize)
    except IOError:
        font = ImageFont.load_default(size=fontsize)

    for ann in bboxes:
        bbox = ann['bbox']
        # draw bbox
        if len(bbox) == 4:
            # draw bbox
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                (xmin, ymin)],
                width=2,
                fill='red')
        else:
            print('the shape of bbox must be [M, 4]')

    for ann in bboxes:
        catname, bbox, score, status, fracture_prob = ann['category_name'], ann['bbox'], ann['score'], ann['status'], ann['fracture_prob']
        xmin, ymin, w, h = bbox
        # draw label
        text = "{} {:.4f} \n {} {:.4f}".format(catname, score, status, fracture_prob)
        # tw, th = draw.textsize(text)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        #label框
        draw.rectangle([(xmin + 1, ymin + 1), (xmin + tw + 1, ymin + th + 1 + 10)], fill='white') 
        # draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill = color)
        # label文字 
        # (xmin + 1, ymin - th)
        if status == "fracture":
            draw.text((xmin + 1, ymin + 1), text, fill='black', font=font) 
        else:
            draw.text((xmin + 1, ymin + 1), text, fill='red', font=font) 

    return image