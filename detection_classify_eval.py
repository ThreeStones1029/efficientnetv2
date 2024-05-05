'''
Description: the file will be used to detection spine xray and then check the vertebral body for fractures
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-12 08:28:55
LastEditors: ShuaiLei
LastEditTime: 2024-05-05 13:51:41
'''
import os
import sys
import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from tools.io.common import load_json_file, save_json_file
from tools.coco.precoco import PreCOCO
from my_dataset import TestDataSet, MyDataSet
from tools.bbox.bbox_process import get_cut_bbox
from tools.vis.bbox_pre_visualize import draw_bbox


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
    
 

def main(args):
    gt_detection_data = COCO(args.gt_bbox_json_file)
    classify_catid2catname = load_json_file('./class_indices.json')
    classify_catname2catid = {catname:catid for catid, catname in classify_catid2catname.items()}
    detection_catid2catname = {}
    for category in gt_detection_data.dataset["categories"]:
        detection_catid2catname[category["id"]] = category["name"]
    test_cut_images, test_cut_images_class, vertebrae_bbox_id_list = get_detection_result_from_gt(args.infer_dir, gt_detection_data, classify_catname2catid, detection_catid2catname)
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
    test_dataset = MyDataSet(images=test_cut_images,
                                images_class=test_cut_images_class,
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
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    sample_num = 0
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]
            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            pred_scores = torch.max(torch.softmax(pred, dim=1), dim=1)[0]
            all_pred_classes += pred_classes.cpu()
            all_pred_scores += pred_scores.cpu()
        eval_acc = accu_num.item() / sample_num

    bboxes_fracture_info = []
    ann_idToann = {}
    for ann in gt_detection_data.dataset["annotations"]:
        ann_idToann[ann["id"]] = ann
    fracture_true_num = 0
    fracture_num = 0 
    normal_true_num = 0
    normal_num = 0
    # The classification prediction results of the cropped vertebral images are saved
    for classify_bbox_id, cut_image in enumerate(test_cut_images):
        # Obtain the bbox_id in the detection of the bbox_id of the current predicted vertebra
        detection_bbox_id = classify_bbox_id2detect_bbox_id[classify_bbox_id]
        predict_class = all_pred_classes[classify_bbox_id].numpy()
        fracture_prob = all_pred_scores[classify_bbox_id].numpy()
        ann = ann_idToann[detection_bbox_id]
        file_name = gt_detection_data.loadImgs(ann["image_id"])[0]["file_name"]
        category_name = detection_catid2catname[ann["category_id"]]
        ann["status"] = classify_catid2catname[str(predict_class)].split("_")[0]
        ann["fracture_prob"] = float(fracture_prob)
        ann["file_name"] = file_name
        ann["category_name"] = category_name
        ann["score"] = 1.0
        bboxes_fracture_info.append(ann)
        if category_name == "fracture":
            fracture_num += 1
            if category_name == ann["status"]:
                fracture_true_num += 1  
        if category_name == "normal":
            normal_num += 1
            if category_name == ann["status"]:
                normal_true_num += 1   
        print("file_name: {}, ann_id: {}, category_name: {}, status: {} fracture_prob: {:.3}".format(file_name, ann["id"], category_name, ann["status"], fracture_prob))
    print("fracture acc is {}".format(fracture_true_num / fracture_num))
    print("normal acc is {}".format(normal_true_num / normal_num))
    print("The overall acc is {}".format(eval_acc))
    # save classify result to json file.
    if args.save_results:
        save_json_file(bboxes_fracture_info, os.path.join(args.output_dir, "bbox.json"))
    # visualize detection and classify result
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
    parser.add_argument('--infer_dir', type=str, default="dataset/spine_fracture/xray/test", help="images infer")
    # detection parameter
    parser.add_argument('--save_cut_images', type=bool, default=True, help="if true, cut images will be saved")
    # Classification paramater
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--weights_category", type=str, default="l", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/spine_fracture/xray/l/val_best_model.pth", help="infer weight path")
    parser.add_argument('--output_dir', type=str, default="infer_output", help="infer image save path")
    parser.add_argument('--visualize', type=bool, default=True, help="whether visualize result")
    parser.add_argument('--save_results', type=bool, default=True, help="whether save detection and fracture result")
    # eval
    parser.add_argument('--gt_bbox_json_file', type=str, default="dataset/spine_fracture/xray/annotations/bbox_test.json", 
                        help="the test images gt json file which record fracture and normal information")
    opt = parser.parse_args()
    main(opt)