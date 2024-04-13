'''
Description: the file will be used to detection spine xray and then check the vertebral body for fractures
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-12 08:28:55
LastEditors: ShuaiLei
LastEditTime: 2024-04-13 09:00:06
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
from tools.io.common import load_json_file
from tools.coco.precoco import PreCOCO
from detection.rtdetr_detection import rtdetr_infer
from detection.yolov5_detection import yolov5_infer
from my_dataset import ValDataSetNoRead


def get_detection_result(infer_dir, infer_image, is_run_detection, chooosed_detection_model, bbox_json_file):
    """
    The function will used to get detection result.
    infer_dir: infer images save folder.
    infer_image: thr infer image.
    is_run_detection: wheather run detection.
    chooosed_detection_model: The detection model.
    bbox_json_file: the detection result.
    """
    imgToAnns = {}
    test_images = []
    imgidToimages = {}
    if is_run_detection == False:
        detection_data = PreCOCO(bbox_json_file)
        imgToAnns = detection_data.imgToAnns
        for img_id, anns in imgToAnns.items():
            print(img_id, anns)
    else:
        if chooosed_detection_model == "rtdetr":
            imgToAnns = rtdetr_infer(infer_dir, infer_image)
        if chooosed_detection_model == "yolov5":
            imgToAnns = yolov5_infer(infer_dir, infer_image)
    return imgToAnns
    


def main(args):
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
    test_images = get_detection_result(args.infer_dir, args.infer_image, args.is_run_detection, args.chooosed_detection_model, args.bbox_json_file)
    # test_dataset = ValDataSetNoRead(images=test_images,
    #                                 transform=data_transform)

    # batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))

    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=batch_size,
    #                                           shuffle=False,
    #                                           pin_memory=True,
    #                                           num_workers=nw,
    #                                           collate_fn=test_dataset.collate_fn)
        
    # # create model
    # if args.weights_category == "s":
    #     model = efficientnetv2_s(num_classes=args.num_classes).to(device)
    # if args.weights_category == "m":
    #     model = efficientnetv2_m(num_classes=args.num_classes).to(device)
    # if args.weights_category == "l":
    #     model = efficientnetv2_l(num_classes=args.num_classes).to(device)
    # # load model weights
    # model.load_state_dict(torch.load(args.model_path, map_location=device))

    # model.eval()
    # data_loader = tqdm(test_loader, file=sys.stdout)
    # all_pred_classes = []
    # all_pred_scores = []
    # with torch.no_grad():
    #     for step, data in enumerate(data_loader):
    #         images = data
    #         pred = model(images.to(device)).cpu()
    #         pred_scores = torch.max(torch.softmax(pred, dim=1), dim=1)[0]
    #         pred_classes = torch.max(pred, dim=1)[1]
    #         all_pred_classes += pred_classes
    #         all_pred_scores += pred_scores

    # class_indict = load_json_file('./class_indices.json')
    # # load image
    # for i, image_path in enumerate(test_images_path):
    #     image = Image.open(image_path)
    #     plt.imshow(image)
    #     predict_class = all_pred_classes[i].numpy()
    #     score = all_pred_scores[i].numpy()
    #     print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)], score)
    #     plt.title(print_res)
    #     plt.savefig(os.path.join(args.infer_output_dir, os.path.basename(image_path)))
    #     print("{}, class: {}   prob: {:.3}".format(os.path.basename(image_path), class_indict[str(predict_class)], score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 测试数据集所在根目录
    parser.add_argument('--infer_dir', type=str, default="dataset/spine_fracture/drr/test", help="multi images infer")
    parser.add_argument('--infer_image', type=str, default="", help="single image infer")
    # detection parameter
    parser.add_argument('--is_run_detection', type=bool, default=False, help="if run detection or not")
    parser.add_argument('--choosed_detection_model', type="yolov5", default=False, help="the detection model")
    parser.add_argument('--save_cut_images', type=bool, default=False, help="save or not save the detection bbox images")
    parser.add_argument('--bbox_json_file', type=str, default="", help="if not run detection, load the detection bbox result json")
    # Classification paramater
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument("--weights_category", type=str, default="s", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/spine_fracture/drr/LA/s/val_best_model.pth", help="infer weight path")
    parser.add_argument('--infer_output_dir', type=str, default="infer_output", help="infer image save path")
    opt = parser.parse_args()
    main(opt)