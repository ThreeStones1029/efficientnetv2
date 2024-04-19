'''
Description: the file will be used to detection spine xray and then check the vertebral body for fractures
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-12 08:28:55
LastEditors: ShuaiLei
LastEditTime: 2024-04-19 04:42:34
'''
import os
import sys
import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from tools.io.common import load_json_file, save_json_file
from tools.coco.precoco import PreCOCO
from detection.rtdetr_detection import rtdetr_paddle_infer, rtdetr_pytorch_infer
from detection.yolov5_detection import yolov5_infer
from my_dataset import TestDataSet
from tools.bbox.bbox_process import get_cut_bbox, filter_low_score_bboxes
from tools.vis.bbox_visualize import draw_bbox


rtdetr_pytorch_infer_parameter = {"envs_path": "/root/anaconda3/bin/python",
                                  "detection_script_path": "/home/RT-DETR/rtdetr_pytorch/tools/infer.py", 
                                  "config_path": "/home/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
                                  "model_path": "/home/RT-DETR/rtdetr_pytorch/output/fracture_dataset/semantic/rtdetr_r50vd_6x_coco/best_checkpoint.pth"}

rtdetr_paddle_infer_parameter = {"envs_path": "/root/anaconda3/envs/rtdetr/bin/python",
                                  "detection_script_path": "/home/RT-DETR/rtdetr_paddle/tools/infer.py", 
                                  "config_path": "/home/RT-DETR/rtdetr_paddle/configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
                                  "model_path": "/home/RT-DETR/rtdetr_paddle/output/fracture_dataset/semantic/rtdetr_r50vd_6x_coco/best_model.pdparams"}

yolov5_infer_parameter = {"envs_path": "",
                          "detection_script_path": "", 
                          "config_path": ""}


def get_detection_result(infer_dir, is_run_detection, detection_model, output_dir, bbox_json_file, threshold):
    """
    The function will used to get detection result.
    infer_dir: infer images save folder.
    infer_image: thr infer image.
    is_run_detection: wheather run detection.
    chooosed_detection_model: The detection model.
    bbox_json_file: the detection result.
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
    cut_images, bbox_id_list = get_cut_images_from_bboxes(infer_dir, bbox_json_file)
    return cut_images, bbox_id_list


def get_cut_images_from_bboxes(infer_dir, bbox_json_file):
    """
    the function will be uesd to cut images for spine fracture classify.
    infer_dir: The infer images.
    param: bbox_json_file: The detection result.
    """
    detection_data = PreCOCO(bbox_json_file)
    imgToAnns = detection_data.imgToAnns
    bbox_id_list = []
    cut_images = []
    for img_id, anns in imgToAnns.items():
        image = Image.open(os.path.join(infer_dir, anns[0]["file_name"])).convert('RGB')
        width, height = image.size
        for ann in anns:
            cut_bbox = get_cut_bbox(ann["bbox"], width, height, expand_coefficient=1.1)
            cut_image = image.crop((cut_bbox[0], cut_bbox[1], cut_bbox[2], cut_bbox[3]))
            bbox_id_list.append(ann["id"])
            cut_images.append(cut_image)
    return cut_images, bbox_id_list
    

def main(args):
    test_cut_images, bbox_id_list = get_detection_result(args.infer_dir, args.is_run_detection, args.detection_model, args.output_dir, args.bbox_json_file, args.draw_threshold)

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

    bboxes_fracture_info = []
    for i, cut_image in enumerate(test_cut_images):
        predict_class = all_pred_classes[i].numpy()
        score = all_pred_scores[i].numpy()
        ann = ann_idToann[i]
        ann["status"] = class_indict[str(predict_class)].split("_")[0]
        ann["fracture_prob"] = float(score)
        bboxes_fracture_info.append(ann)
        print("file_name: {}, ann_id: {}, category_name: {}, status: {} fracture_prob: {:.3}".format(ann["file_name"], ann["id"], ann["category_name"], class_indict[str(predict_class)], score))
    if args.save_results:
        save_json_file(bboxes_fracture_info, os.path.join(args.output_dir, "bbox.json"))
    if args.visualize:
        detection_classify_data = PreCOCO(bboxes_fracture_info)
        img_idToFilename = detection_classify_data.img_idToFilename
        imid2path = {img_id: os.path.join(args.infer_dir, file_name) for img_id, file_name in img_idToFilename.items()}
        imgToAnns = detection_classify_data.imgToAnns
        for image_id in tqdm(imid2path.keys(), desc="vis bbox"):
        # PIL默认读取为灰度图
            image = Image.open(imid2path[image_id]).convert('RGB')
            bboxes = imgToAnns[image_id]
            vis_image = draw_bbox(image, bboxes)
            vis_image.save(os.path.join(args.output_dir, os.path.basename(imid2path[image_id])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 测试数据集所在根目录
    parser.add_argument('--infer_dir', type=str, default="dataset/spine_fracture/test_drr", help="images infer")
    # detection parameter
    parser.add_argument('--is_run_detection', type=bool, default=True, help="if run detection or not")
    parser.add_argument('--detection_model', type=str, default="rtdetr_pytorch", help="the detection model")
    parser.add_argument('--save_cut_images', type=bool, default=False, help="save or not save the detection bbox images")
    parser.add_argument('--bbox_json_file', type=str, default="infer_output/bbox.json", help="if not run detection, load the detection bbox result json")
    parser.add_argument('--draw_threshold', type=str, default=0.6, help="the threshold used to filter bbox and visualize")
    # Classification paramater
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--weights_category", type=str, default="l", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/spine_fracture/drr/all/l/val_best_model.pth", help="infer weight path")
    parser.add_argument('--output_dir', type=str, default="infer_output", help="infer image save path")
    parser.add_argument('--visualize', type=bool, default=True, help="whether visualize result")
    parser.add_argument('--save_results', type=bool, default=True, help="whether save detection and fracture result")
    opt = parser.parse_args()
    main(opt)