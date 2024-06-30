'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-06-14 02:13:53
LastEditors: ShuaiLei
LastEditTime: 2024-06-14 07:34:43
'''
import os
import json
import cv2

def yolo_to_coco(input_dir, output_file):
    images = []
    annotations = []
    categories = []
    category_set = set()
    annotation_id = 1
    catid2catname = {0: "person",
                    1: "bicycle",
                    2: "car",
                    3: "motorcycle",
                    4: "airplane",
                    5: "bus",
                    6: "train",
                    7: "truck",
                    8: "boat",
                    9: "traffic light",
                    10: "fire hydrant",
                    11: "stop sign",
                    12: "parking meter",
                    13: "bench",
                    14: "bird",
                    15: "cat",
                    16: "dog",
                    17: "horse",
                    18: "sheep",
                    19: "cow",
                    20: "elephant",
                    21: "bear",
                    22: "zebra",
                    23: "giraffe",
                    24: "backpack",
                    25: "umbrella",
                    26: "handbag",
                    27: "tie",
                    28: "suitcase",
                    29: "frisbee",
                    30: "skis",
                    31: "snowboard",
                    32: "sports ball",
                    33: "kite",
                    34: "baseball bat",
                    35: "baseball glove",
                    36: "skateboard",
                    37: "surfboard",
                    38: "tennis racket",
                    39: "bottle",
                    40: "wine glass",
                    41: "cup",
                    42: "fork",
                    43: "knife",
                    44: "spoon",
                    45: "bowl",
                    46: "banana",
                    47: "apple",
                    48: "sandwich",
                    49: "orange",
                    50: "broccoli",
                    51: "carrot",
                    52: "hot dog",
                    53: "pizza",
                    54: "donut",
                    55: "cake",
                    56: "chair",
                    57: "couch",
                    58: "potted plant",
                    59: "bed",
                    60: "dining table",
                    61: "toilet",
                    62: "tv",
                    63: "laptop",
                    64: "mouse",
                    65: "remote",
                    66: "keyboard",
                    67: "cell phone",
                    68: "microwave",
                    69: "oven",
                    70: "toaster",
                    71: "sink",
                    72: "refrigerator",
                    73: "book",
                    74: "clock",
                    75: "vase",
                    76: "scissors",
                    77: "teddy bear",
                    78: "hair drier",
                    79: "toothbrush"}

    image_dir = os.path.join(input_dir, "images", "train2017")
    label_dir = os.path.join(input_dir, "labels", "train2017")

    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image_id = int(image_file.split(".")[0].lstrip('0'))
        
        # Read image to get its width and height
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        label_file = image_file.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue

        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_file
        })
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])
            
            # Convert from relative YOLO format to absolute COCO format
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width = bbox_width * width
            bbox_height = bbox_height * height
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            annotation_id += 1
            
            category_set.add(class_id)
    
    # Create category list
    for category_id in sorted(category_set):
        categories.append({
            "id": category_id,
            "name": catid2catname[category_id]
        })
    
    # Create the final COCO JSON structure
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {}
    }
    
    # Write COCO JSON to file
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Usage example
input_dir = "/home/ABLSpineLevelCheck_single/datasets/coco128"
output_file = "/home/ABLSpineLevelCheck_single/datasets/coco128/annotations/train2017.json"
yolo_to_coco(input_dir, output_file)


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval(gt_json, pre_json):
    anno = COCO(gt_json)  # init annotations api
    pred = anno.loadRes(pre_json)  # init predictions api
    eval = COCOeval(anno, pred, "bbox")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

eval(output_file, "/home/ABLSpineLevelCheck_single/yolov5/runs/detect/coco128/bbox.json")
eval(output_file, "/home/ABLSpineLevelCheck_single/yolov5/runs/val/coco128/yolov5s_predictions.json")



