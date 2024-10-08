'''
Description: the file will be used to detection spine xray and then check the vertebral body for fractures
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-12 08:28:55
LastEditors: ShuaiLei
LastEditTime: 2024-09-09 03:48:55
'''
import os
import sys
import argparse
import torch
import itertools
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tools.coco.precoco import PreCOCO
from tqdm import tqdm
from PIL import Image
import numpy as np
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from tools.io.common import load_json_file, save_json_file
from my_dataset import TestDataSet
from tools.vis.bbox_pre_visualize import draw_bbox
from detection.rtdetr_detection import rtdetr_paddle_infer, rtdetr_pytorch_infer
from detection.yolov5_detection import yolov5_infer
from tools.bbox.bbox_process import get_cut_bbox, filter_low_score_bboxes


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
                                  "model_path": "/home/RT-DETR/rtdetr_paddle/output/TD20240705_LA/fold1/semantic/rtdetr_r50vd_6x_coco/best_model.pdparams"}

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


def eval(truth_bbox_json_path, predict_bbox_json_path, classwise):
    # with open(predict_bbox_json_path, "r") as f:
    #     predict = json.load(f)
    #     filter_dataset = []
    # for ann in predict:
    #     if ann["score"] > 0.5:
    #         filter_dataset.append(ann)
    # over_0_5_json_path = os.path.abspath(os.path.join(os.path.dirname(predict_bbox_json_path), "result_over0_5.json"))
    # with open(over_0_5_json_path, "w") as f:
    #     json.dump(filter_dataset, f)

    coco_gt = COCO(truth_bbox_json_path)
    coco_dt = coco_gt.loadRes(predict_bbox_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            print('terminaltables not found, plaese install terminaltables.', 'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision_ap50 = precisions[0, :, idx, 0, -1]
            precision_ap75 = precisions[5, :, idx, 0, -1]
            precision = precision[precision > -1]
            precision_ap50 = precision_ap50[precision_ap50 > -1]
            precision_ap75 = precision_ap75[precision_ap75 > -1]
            if precision.size:
                ap = np.mean(precision)
                ap50 = np.mean(precision_ap50)
                ap75 = np.mean(precision_ap75)
            else:
                ap = float('nan')
            results_per_category.append((str(nm["name"]), '{:0.3f}'.format(float(ap)), '{:0.3f}'.format(float(ap50)), '{:0.3f}'.format(float(ap75))))
            pr_array = precisions[0, :, idx, 0, 2]
            recall_array = np.arange(0.0, 1.01, 0.01)

        num_columns = min(8, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP', 'AP50', 'AP75'] * (num_columns // 4)
        results_2d = itertools.zip_longest(* [results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('Per-category of {} AP: \n{}'.format("bbox", table.table))


def main(args):
    test_cut_images, vertebrae_bbox_id_list = get_detection_result(args.infer_dir, 
                                                                   args.is_run_detection, 
                                                                   args.detection_model, 
                                                                   args.output_dir, 
                                                                   args.save_cut_images,
                                                                   args.bbox_expand_coefficient,
                                                                   args.bbox_json_file, 
                                                                   args.draw_threshold)
    
    classify_bbox_id2detect_bbox_id = {i: vertebrae_bbox_id for i, vertebrae_bbox_id in enumerate(vertebrae_bbox_id_list)}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = args.weights_category
    data_transform = transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                         transforms.CenterCrop(img_size[num_model][1]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # 实例化验证数据集
    test_dataset = TestDataSet(images=test_cut_images,
                                transform=data_transform)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)
        
    # create model
    if args.weights_category == "s":
        model = efficientnetv2_s(num_classes=args.num_classes).to(device)
    if args.weights_category == "m":
        model = efficientnetv2_m(num_classes=args.num_classes).to(device)
    if args.weights_category == "l":
        model = efficientnetv2_l(num_classes=args.num_classes).to(device)
    # load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    model.eval()
    data_loader = tqdm(test_loader, file=sys.stdout)
    all_pred_classes = []
    all_pred_scores = []
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images = data
            pred = model(images.to(device)).cpu()
            pred_scores = torch.max(torch.softmax(pred, dim=1), dim=1)[0]
            pred_classes = torch.max(pred, dim=1)[1]
            all_pred_classes += pred_classes
            all_pred_scores += pred_scores

    class_indict = load_json_file('./class_indices.json')
    detection_data = PreCOCO(args.bbox_json_file)
    ann_idToann = detection_data.ann_idToann
    detection_catname2catid = {"normal": 1, "fracture": 2}
    bboxes_fracture_info = []
    for classify_bbox_id, cut_image in enumerate(test_cut_images):
        # get bbox id in detection result
        detection_bbox_id = classify_bbox_id2detect_bbox_id[classify_bbox_id]
        predict_class = all_pred_classes[classify_bbox_id].numpy()
        score = all_pred_scores[classify_bbox_id].numpy()
        ann = ann_idToann[detection_bbox_id]
        # ann["status"] = class_indict[str(predict_class)].split("_")[0]
        # ann["fracture_prob"] = float(score)
        ann["category_name"] = class_indict[str(predict_class)].split("_")[0]
        ann["category_id"] = detection_catname2catid[ann["category_name"]]
        ann["score"] = float(score)
        bboxes_fracture_info.append(ann)
        print("file_name: {}, ann_id: {}, category_name: {}, score: {}".format(ann["file_name"], ann["id"], ann["category_name"], score))
    if args.save_results:
        save_json_file(bboxes_fracture_info, os.path.join(args.output_dir, "bbox.json"))
    if args.is_eval:
        eval(args.gt_json_file, os.path.join(args.output_dir, "bbox.json"), classwise=True)
    if args.visualize:
        detection_classify_data = PreCOCO(bboxes_fracture_info)
        img_idToFilename = detection_classify_data.img_idToFilename
        imid2path = {img_id: os.path.join(args.infer_dir, file_name) for img_id, file_name in img_idToFilename.items()}
        imgToAnns = detection_classify_data.imgToAnns
        for image_id in tqdm(imid2path.keys(), desc="vis bbox"):
        # PIL默认读取为灰度图
            image = Image.open(imid2path[image_id]).convert('RGB')
            bboxes = imgToAnns[image_id]
            vis_image = draw_bbox(image, bboxes, fontsize=30)
            vis_image.save(os.path.join(args.output_dir, os.path.basename(imid2path[image_id])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 测试数据集所在根目录
    parser.add_argument('--infer_dir', type=str, default="/home/RT-DETR/rtdetr_paddle/datasets/TD20240705_LA/split_dataset/fold1/val", help="images infer")
    # detection parameter
    parser.add_argument('--is_run_detection', type=bool, default=True, help="if run detection or not")
    parser.add_argument('--detection_model', type=str, default="rtdetr_paddle", help="the detection model")
    parser.add_argument('--bbox_json_file', type=str, default="infer_output/bbox.json", help="if not run detection, load the detection bbox result json")
    parser.add_argument('--save_cut_images', type=bool, default=True, help="if true, cut images will be saved")
    parser.add_argument('--draw_threshold', type=float, default=0.5, help="the threshold used to filter bbox and visualize")
    # Classification paramater
    parser.add_argument('--bbox_expand_coefficient', type=float, default=1.5, help="the cut bbox expand_coefficient")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--weights_category", type=str, default="m", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/TD20240705_LA/preoperative_pretrain/fold1/m/val_best_model.pth", help="infer weight path")
    parser.add_argument('--output_dir', type=str, default="infer_output", help="infer image save path")
    parser.add_argument('--visualize', type=bool, default=True, help="whether visualize result")
    parser.add_argument('--save_results', type=bool, default=True, help="whether save detection and fracture result")
    # detection and classify eval
    parser.add_argument('--is_eval', type=bool, default=True, help="whether eval results")
    parser.add_argument('--gt_json_file', type=str, default="dataset/TD20240705_LA/split_dataset/fold1/annotations/fracture_bbox_val.json", help="the fracture detection json file")
    opt = parser.parse_args()
    main(opt)