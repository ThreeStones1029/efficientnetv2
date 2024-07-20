'''
Description: the file will be used to detection spine xray and then check the vertebral body for fractures
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-12 08:28:55
LastEditors: ShuaiLei
LastEditTime: 2024-07-20 02:59:05
'''
import os
import sys
import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tools.coco.precoco import PreCOCO
from tqdm import tqdm
from PIL import Image
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from tools.io.common import load_json_file, save_json_file
from my_dataset import TestDataSet
from tools.vis.bbox_pre_visualize import draw_bbox
from get_detection_gt_or_predict_bbox import get_detection_result


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

    bboxes_fracture_info = []
    for classify_bbox_id, cut_image in enumerate(test_cut_images):
        # get bbox id in detection result
        detection_bbox_id = classify_bbox_id2detect_bbox_id[classify_bbox_id]
        predict_class = all_pred_classes[classify_bbox_id].numpy()
        score = all_pred_scores[classify_bbox_id].numpy()
        ann = ann_idToann[detection_bbox_id]
        ann["status"] = class_indict[str(predict_class)].split("_")[0]
        ann["fracture_prob"] = float(score)
        bboxes_fracture_info.append(ann)
        print("file_name: {}, ann_id: {}, category_name: {}, status: {} fracture_prob: {:.3}".format(ann["file_name"], ann["id"], ann["category_name"], class_indict[str(predict_class)], score))
    # add rib pelvis and bone_cement into bboxes_fracture_info
    for ann in detection_data.dataset["annotations"]:
        if ann["category_name"] != "vertebrae":
            ann["status"] = "not vertebrae"
            ann["fracture_prob"] = 0
            bboxes_fracture_info.append(ann)
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
            vis_image = draw_bbox(image, bboxes, fontsize=30)
            vis_image.save(os.path.join(args.output_dir, os.path.basename(imid2path[image_id])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 测试数据集所在根目录
    parser.add_argument('--infer_dir', type=str, default="/home/RT-DETR/rtdetr_paddle/datasets/Fracture_dataset/test", help="images infer")
    # detection parameter
    parser.add_argument('--is_run_detection', type=bool, default=True, help="if run detection or not")
    parser.add_argument('--detection_model', type=str, default="rtdetr_paddle", help="the detection model")
    parser.add_argument('--bbox_json_file', type=str, default="infer_output1/bbox.json", help="if not run detection, load the detection bbox result json")
    parser.add_argument('--save_cut_images', type=bool, default=True, help="if true, cut images will be saved")
    parser.add_argument('--draw_threshold', type=float, default=0.6, help="the threshold used to filter bbox and visualize")
    # Classification paramater
    parser.add_argument('--bbox_expand_coefficient', type=float, default=1.5, help="the cut bbox expand_coefficient")
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--weights_category", type=str, default="s", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/spine_fracture/drr/s/val_best_model.pth", help="infer weight path")
    parser.add_argument('--output_dir', type=str, default="infer_output1", help="infer image save path")
    parser.add_argument('--visualize', type=bool, default=True, help="whether visualize result")
    parser.add_argument('--save_results', type=bool, default=True, help="whether save detection and fracture result")
    opt = parser.parse_args()
    main(opt)