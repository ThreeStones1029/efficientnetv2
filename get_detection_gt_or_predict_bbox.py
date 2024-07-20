import os
from PIL import Image
from tools.bbox.bbox_process import get_cut_bbox
from detection.rtdetr_detection import rtdetr_paddle_infer, rtdetr_pytorch_infer
from detection.yolov5_detection import yolov5_infer
from tools.bbox.bbox_process import get_cut_bbox, filter_low_score_bboxes
from tools.coco.precoco import PreCOCO


def get_detection_result_from_gt(infer_dir, gt_detection_data, classify_catname2catid, detection_catid2catname):
    """
    the function will be used to get truth labels about fracture status in test images vertebraes.
    param: infer_dir: The infer images.
    param: gt_detection_data: the gt bboxes.
    param: classify_catname2catid: the classify category name to category id dict.
    """
    imgToAnns = gt_detection_data.imgToAnns
    # record vertebrae bbox id
    vertebrae_bbox_id_list = []
    cut_images_list = []
    cut_images_classify_label_list = []
    for img_id, anns in imgToAnns.items():
        file_name = gt_detection_data.loadImgs(img_id)[0]["file_name"]
        image = Image.open(os.path.join(infer_dir, file_name)).convert('RGB')
        width, height = image.size
        for ann in anns:
            cut_bbox = get_cut_bbox(ann["bbox"], width, height, expand_coefficient=1.5)
            cut_image = image.crop((cut_bbox[0], cut_bbox[1], cut_bbox[2], cut_bbox[3]))
            vertebrae_bbox_id_list.append(ann["id"])
            cut_images_list.append(cut_image)
            detection_catname = detection_catid2catname[ann["category_id"]]
            classify_catname = detection_catname + "_images"
            cut_images_classify_label_list.append(int(classify_catname2catid[classify_catname]))
    return cut_images_list, cut_images_classify_label_list, vertebrae_bbox_id_list


# drr
# rtdetr_pytorch_infer_parameter = {"envs_path": "/root/anaconda3/bin/python",
#                                   "detection_script_path": "/home/RT-DETR/rtdetr_pytorch/tools/infer.py", 
#                                   "config_path": "/home/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
#                                   "model_path": "/home/RT-DETR/rtdetr_pytorch/output/fracture_dataset/semantic/rtdetr_r50vd_6x_coco/best_checkpoint.pth"}

# rtdetr_paddle_infer_parameter = {"envs_path": "/root/anaconda3/envs/rtdetr/bin/python",
#                                   "detection_script_path": "/home/RT-DETR/rtdetr_paddle/tools/infer.py", 
#                                   "config_path": "/home/RT-DETR/rtdetr_paddle/configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
#                                   "model_path": "/home/RT-DETR/rtdetr_paddle/output/fracture_dataset/semantic/rtdetr_r50vd_6x_coco/best_model.pdparams"}

# xray
rtdetr_pytorch_infer_parameter = {"envs_path": "/root/anaconda3/bin/python",
                                  "detection_script_path": "/home/RT-DETR/rtdetr_pytorch/tools/infer.py", 
                                  "config_path": "/home/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
                                  "model_path": "/home/RT-DETR/rtdetr_pytorch/output/Fracture_dataset/semantic/rtdetr_r50vd_6x_coco/best_checkpoint.pth"}

rtdetr_paddle_infer_parameter = {"envs_path": "/root/anaconda3/envs/rtdetr/bin/python",
                                  "detection_script_path": "/home/RT-DETR/rtdetr_paddle/tools/infer.py", 
                                  "config_path": "detection/rtdetr_paddle_configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
                                  "model_path": "/home/RT-DETR/rtdetr_paddle/output/Fracture_dataset/semantic/rtdetr_r50vd_6x_coco/best_model.pdparams"}

yolov5_infer_parameter = {"envs_path": "",
                          "detection_script_path": "", 
                          "config_path": ""}


def get_detection_result(infer_dir, 
                         is_run_detection, 
                         detection_model, 
                         output_dir, 
                         save_cut_images,
                         bbox_expand_coefficient,
                         bbox_json_file, 
                         threshold):
    """
    The function will used to get detection result.
    param: infer_dir: infer images save folder.
    param: is_run_detection: wheather run detection.
    param: detection_model: The detection model.
    param: output_dir: The detection result save path.
    param: save_cut_images: whether save cut_images.
    param: bbox_json_file: the detection result.
    param: threshold: the score threshold will be used to filter bbox.
    """
    if is_run_detection == True:
        assert detection_model in ["rtdetr_paddle", "rtdetr_pytorch", "yolov5"], 'detection model {} not supported'.format(detection_model)
        if detection_model == "rtdetr_paddle":
            rtdetr_paddle_infer(rtdetr_paddle_infer_parameter, infer_dir, output_dir)
        if detection_model == "rtdetr_pytorch":
            rtdetr_pytorch_infer(rtdetr_pytorch_infer_parameter, infer_dir, output_dir)
        if detection_model == "yolov5":
            yolov5_infer(yolov5_infer_parameter, infer_dir, output_dir)
    filter_low_score_bboxes(bbox_json_file, threshold) 
    cut_images_list, bbox_id_list = get_cut_images_from_pre_bboxes(infer_dir, bbox_json_file, save_cut_images, bbox_expand_coefficient)
    return cut_images_list, bbox_id_list


def get_cut_images_from_pre_bboxes(infer_dir, bbox_json_file, save_cut_images, bbox_expand_coefficient):
    """
    the function will be uesd to cut images for spine fracture classify.
    param: infer_dir: The infer images.
    param: bbox_json_file: The detection result.
    param: save_cut_images: whether save cut_images.
    """
    pre_detection_data = PreCOCO(bbox_json_file)
    output_dir = os.path.dirname(bbox_json_file)
    imgToAnns = pre_detection_data.imgToAnns
    # record vertebrae bbox id
    vertebrae_bbox_id_list = []
    cut_images_list = []
    for img_id, anns in imgToAnns.items():
        i = 0
        file_name = anns[0]["file_name"]
        image = Image.open(os.path.join(infer_dir, file_name)).convert('RGB')
        width, height = image.size
        for ann in anns:
            if ann["category_name"] == "vertebrae":
                i += 1
                cut_bbox = get_cut_bbox(ann["bbox"], width, height, bbox_expand_coefficient)
                cut_image = image.crop((cut_bbox[0], cut_bbox[1], cut_bbox[2], cut_bbox[3]))
                if save_cut_images:
                    os.makedirs(os.path.join(output_dir, "cut"), exist_ok=True)
                    cut_image.save(os.path.join(output_dir, "cut",file_name + "_" + str(i) + ".png"))
                vertebrae_bbox_id_list.append(ann["id"])
                cut_images_list.append(cut_image)
    return cut_images_list, vertebrae_bbox_id_list