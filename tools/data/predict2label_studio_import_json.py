'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-06-19 14:17:03
LastEditors: ShuaiLei
LastEditTime: 2024-07-10 08:27:25
'''
import os
import sys
current_file_path = os.path.abspath(__file__)
project_root = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from tools.io.common import load_json_file, save_json_file
from tools.coco.precoco import PreCOCO
from PIL import Image


def predict2label_studio_import(predict_json_file, images_folder, label_studio_import_json_file):
    """
    用于将预测的结果转为label_studio格式的标注,作为真实标注导入label_studio
    param:predict_json_file
    param:label_studio_annotation_file
    param:images_folder
    """
    gt_data = []
    predict = PreCOCO(predict_json_file)
    images_folder_name = images_folder.split("/")[-1]
    cur_image_id = 0
    cur_annotation_id = 0
    cur_import_id = 0
    
    for image_id, anns in predict.imgToAnns.items():
        cur_image_id += 1
        cur_annotation_id += 1
        cur_import_id += 1
        single_data = {"id": cur_image_id,
                    "data": {"img": "",
                            "L4L6": "need check",
                            "type": "LA",
                            "Pixel Spacing": "None"},
                    "annotations": [
                    {
                        "id": cur_annotation_id,
                        "created_username": " bot@prmlk.com, 3",
                        "created_ago": "1 minutes",
                        "completed_by": {
                        "id": 3,
                        "first_name": "",
                        "last_name": "",
                        "avatar": None,
                        "email": "bot@prmlk.com",
                        "initials": "bo"
                        },
                        "result": [],
                        "was_cancelled": False,
                        "ground_truth": True,
                        "created_at": "2024-07-10T12:38:08.667557Z",
                        "updated_at": "2024-07-10T12:38:08.667577Z",
                        "draft_created_at": None,
                        "lead_time": None,
                        "import_id": cur_import_id,
                        "last_action": None,
                        "task": cur_image_id,
                        "project": 15,
                        "updated_by": None,
                        "parent_prediction": None,
                        "parent_annotation": None,
                        "last_created_by": None
                    }
                    ],
                    "predictions": []
                }
        file_name = predict.img_idToFilename[image_id]
        single_data["data"]["img"] = "/data/local-files/?d=" + images_folder_name + "/" + file_name
        image = Image.open(os.path.join(images_folder, file_name)).convert('RGB')
        image_width = image.size[0]
        image_height = image.size[1]
        for ann in anns:
            single_result = {"type": "rectanglelabels",
                         "to_name": "img-1",
                         "from_name": "bbox",
                         "image_rotation": 0,
                         "original_width": image_width,
                         "original_height": image_height}
            if ann["score"] > 0.3:
                value = {"rotation": 0, 
                        "x": ann["bbox"][0] / image_width * 100, 
                        "y": ann["bbox"][1] / image_height * 100,
                        "width": ann["bbox"][2] / image_width * 100, 
                        "height": ann["bbox"][3] / image_height * 100,
                        "rectanglelabels": [ann["category_name"]]}
                single_result["value"] = value
                single_data["annotations"][0]["result"].append(single_result)
        gt_data.append(single_data)
    save_json_file(gt_data, label_studio_import_json_file)
           

if __name__ == "__main__":
   predict2label_studio_import("dataset/spine_fracture/TD_fracture_18_clahe_logic_predict.json",
                               "dataset/spine_fracture/TD_fracture_18_clahe",
                               "dataset/spine_fracture/TD_fracture_18_clahe_label_studio_import.json")