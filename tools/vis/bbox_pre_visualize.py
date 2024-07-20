'''
Description: this function will be used to visualize fracture information and bbox
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-18 13:35:20
LastEditors: ShuaiLei
LastEditTime: 2024-07-20 13:08:19
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
        catname, bbox, score = ann['category_name'], ann['bbox'], ann['score']
        xmin, ymin, w, h = bbox
        # draw label
        text = "{} {:.4f}".format(catname, score)
        # tw, th = draw.textsize(text)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        #label框
        draw.rectangle([(xmin + w, ymin), (xmin + tw + w, ymin + th + 10)], fill='white') 
        # draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill = color)
        # label文字 
        # (xmin + 1, ymin - th)
        draw.text((xmin + w, ymin), text, fill='red', font=font) 

    return image