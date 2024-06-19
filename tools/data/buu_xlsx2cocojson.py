'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-06-19 07:20:09
LastEditors: ShuaiLei
LastEditTime: 2024-06-19 14:04:31
'''
import pandas as pd
import os
import json
from PIL import Image


def xlsx2label_studio(xlsx_annotation_file, label_studio_annotation_file, images_folder=None):
    """
    这个函数用于将BUU数据集的xlsx标注格式转为label_studio格式,导入label_studio中
    param: xlsx_annotation_file: buu数据集原本标注
    param: label_studio_annotation_file: label_studio需要的格式
    param: images_folder 需要转换的图片
    """
    # 读取指定工作表
    gt_data = []
    LA_df = pd.read_excel(xlsx_annotation_file, sheet_name='Pos_LA')
    L4L6_df = pd.read_excel(xlsx_annotation_file, sheet_name='L4L6')
    pixelspacing_df = pd.read_excel(xlsx_annotation_file, sheet_name='Pixel Spacing')
    columns_to_extract = ["L1a_1c", "L1a_1r", "L1a_2c", "L1a_2r", "L1b_1c", "L1b_1r", "L1b_2c", "L1b_2r",
                          "L2a_1c", "L2a_1r", "L2a_2c", "L2a_2r", "L2b_1c", "L2b_1r", "L2b_2c", "L2b_2r",
                          "L3a_1c", "L3a_1r", "L3a_2c", "L3a_2r", "L3b_1c", "L3b_1r", "L3b_2c", "L3b_2r",
                          "L4a_1c", "L4a_1r", "L4a_2c", "L4a_2r", "L4b_1c", "L4b_1r", "L4b_2c", "L4b_2r",
                          "L5a_1c", "L5a_1r", "L5a_2c", "L5a_2r", "L5b_1c", "L5b_1r", "L5b_2c", "L5b_2r"]
    file_name2type = {L4L6_df.loc[i, "filename"] : L4L6_df.loc[i, "L4L6"] for i in range(len(L4L6_df))}
    file_name2pixelspacing = {pixelspacing_df.loc[i, "filename"]: pixelspacing_df.loc[i, "pixel_spacing"] for i in range(len(pixelspacing_df))}  
    image_id = 1
    annotation_id = 1
    import_id = 1
    for i in range(len(LA_df)):
        single_data = {"id": 1,
                    "data": {"img": "",
                            "L4L6": "none",
                            "type": "LA",
                            "Pixel Spacing": "[0.140000, 0.140000]"},
                    "annotations": [
                    {
                        "id": 1,
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
                        "created_at": "2024-06-19T12:38:08.667557Z",
                        "updated_at": "2024-06-19T12:38:08.667577Z",
                        "draft_created_at": None,
                        "lead_time": None,
                        "import_id": 1,
                        "last_action": None,
                        "task": 1,
                        "project": 15,
                        "updated_by": None,
                        "parent_prediction": None,
                        "parent_annotation": None,
                        "last_created_by": None
                    }
                    ],
                    "predictions": []
                }
        if isinstance(LA_df.loc[i, "filename"], str) and LA_df.loc[i, "filename"] + ".jpg" in os.listdir(images_folder):
            single_data["id"] = image_id
            single_data["annotations"][0]["id"] = annotation_id
            single_data["annotations"][0]["import_id"] = import_id
            single_data["annotations"][0]["task"] = image_id
            single_data["data"]["type"] = file_name2type[LA_df.loc[i, "filename"]] if LA_df.loc[i, "filename"] in file_name2type else None
            single_data["data"]["Pixel Spacing"] = file_name2pixelspacing[LA_df.loc[i, "filename"][:-1]]
            file_name = LA_df.loc[i, "filename"] + ".jpg"
            image = Image.open(os.path.join(images_folder, file_name)).convert('RGB')
            single_data["data"]["img"] = "/data/local-files/?d=LA_preoperative_xray_fracture/BUU/" + file_name
            image_width = image.size[0]
            image_height = image.size[1]
            row_data = LA_df.loc[i, columns_to_extract]
            for j in range(0, len(row_data), 8):
                single_result = {"type": "rectanglelabels",
                                 "to_name": "img-1",
                                 "from_name": "bbox",
                                 "image_rotation": 0,
                                 "original_width": image_width,
                                 "original_height": image_height}
                four_points = row_data[j : j + 8]
                four_points_list = four_points.to_list()
                four_points_dict = four_points.to_dict()
                min_x = min(four_points_list[0], four_points_list[2], four_points_list[4], four_points_list[6])
                min_y = min(four_points_list[1], four_points_list[3], four_points_list[5], four_points_list[7])
                max_x = max(four_points_list[0], four_points_list[2], four_points_list[4], four_points_list[6])
                max_y = max(four_points_list[1], four_points_list[3], four_points_list[5], four_points_list[7])
                width = max_x - min_x
                height = max_y - min_y
                category_name = list(four_points_dict.keys())[0][:2]
                value = {"rotation": 0, 
                         "x": min_x / image_width * 100, 
                         "y": min_y / image_height * 100,
                         "width": width / image_width * 100, 
                         "height": height / image_height * 100,
                          "rectanglelabels": [category_name]}
                single_result["value"] = value
                single_data["annotations"][0]["result"].append(single_result)
            gt_data.append(single_data)
            image_id += 1
            annotation_id += 1
            import_id += 1

    with open(label_studio_annotation_file, "w") as f:
        json.dump(gt_data, f)
            

if __name__ == "__main__":
    xlsx2label_studio("dataset/spine_fracture/KIOM2022_dataset_report.xlsx",
                      "dataset/spine_fracture/label_studio_import.json", 
                      "dataset/spine_fracture/LA_preoperative_xray_fracture/BUU")