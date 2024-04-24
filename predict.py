'''
Description: The file will be used to predict category of vertebrae.
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-03-31 04:04:02
LastEditors: ShuaiLei
LastEditTime: 2024-04-24 12:18:23
'''
import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from my_dataset import TestDataSet
from tqdm import tqdm
import sys
import glob
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from tools.io.common import load_json_file


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, "--infer_img or --infer_dir should be set"
    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]
    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), "infer_dir {} is not a directory".format(infer_dir)
    exts = ["jpg", "png"]
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, "no image found in {}".format(infer_dir)
    return images


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
    test_images_path = get_test_images(args.infer_dir, args.infer_image)
    test_dataset = TestDataSet(images_path=test_images_path,
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
    # load image
    for i, image_path in enumerate(test_images_path):
        image = Image.open(image_path)
        plt.imshow(image)
        predict_class = all_pred_classes[i].numpy()
        score = all_pred_scores[i].numpy()
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)], score)
        plt.title(print_res)
        plt.savefig(os.path.join(args.output_dir, os.path.basename(image_path)))
        print("{}, class: {}   prob: {:.3}".format(os.path.basename(image_path), class_indict[str(predict_class)], score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    # 数据集所在根目录
    parser.add_argument('--infer_dir', type=str, default="dataset/spine_fracture/test1", help="multi images infer")
    parser.add_argument('--infer_image', type=str, default="", help="single image infer")
    parser.add_argument("--weights_category", type=str, default="l", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/spine_fracture/drr/all/l/val_best_model.pth", help="infer weight path")
    parser.add_argument('--output_dir', type=str, default="infer_output", help="infer image save path")
    opt = parser.parse_args()
    main(opt)
