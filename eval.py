'''
Description: The function will be used eval classify result.
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-02 13:58:48
LastEditors: ShuaiLei
LastEditTime: 2024-04-19 08:49:54
'''
import os
import sys
import argparse
import torch
from torchvision import transforms
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
from my_dataset import MyDataSet
from utils import evaluate
from tools.io.common import load_json_file
from tools.data.dataset_process import read_images_and_labels_from_txt
from tqdm import tqdm
import random


def get_eval_images_and_labels(data_root, json_file):
    """
    The function will be used to generate categories information and save it in txt file.
    """
    random.seed(0)
    supported_exts = ["jpg", "JPG", "png", "PNG"]
    images_folder_list = [os.path.join(data_root, cla) for cla in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, cla))]
    class_indict = load_json_file(json_file)
    classname2id = {classname: classid for classid, classname in class_indict.items()}
    eval_image_paths = []
    eval_labels = []
    for images_folder in images_folder_list:
        for file_name in os.listdir(images_folder):
            if file_name.split(".")[-1] in supported_exts:
                classname = os.path.basename(images_folder)
                eval_image_paths.append(os.path.join(images_folder, file_name))
                eval_labels.append(int(classname2id[classname]))
    return eval_image_paths, eval_labels


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    assert args.eval_dir != None or args.eval_txt != None, "eval dir and eval txt can not both none."
    if args.eval_dir:
        eval_images_path, eval_images_label = get_eval_images_and_labels(args.eval_dir, args.class_indict_file)
    if args.eval_txt:
        eval_images_path, eval_images_label = read_images_and_labels_from_txt(args.eval_txt)
    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = args.weights_category
    data_transform = transforms.Compose([transforms.Resize(img_size[num_model][1]),
                     transforms.CenterCrop(img_size[num_model][1]),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # 评估数据集
    eval_dataset = MyDataSet(images_path=eval_images_path,
                             images_class=eval_images_label,
                             transform=data_transform)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=eval_dataset.collate_fn)
    # 如果存在预训练权重则载入,根据类型导入[s, m, l]
    if args.weights_category == "s":
        model = efficientnetv2_s(num_classes=args.num_classes).to(device)
    if args.weights_category == "m":
        model = efficientnetv2_m(num_classes=args.num_classes).to(device)
    if args.weights_category == "l":
        model = efficientnetv2_l(num_classes=args.num_classes).to(device)
    # load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    # eval
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    data_loader = tqdm(eval_loader, file=sys.stdout)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]
            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
            loss = loss_function(pred, labels.to(device))
            accu_loss += loss
    eval_loss = accu_loss.item() / (step + 1)
    eval_acc = accu_num.item() / sample_num
    print("[*] The best model acc is {}".format(eval_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    # 数据集所在根目录
    parser.add_argument('--eval_dir', type=str, default=None, help="eval images folder directory")
    parser.add_argument('--eval_txt', type=str, default="dataset/spine_fracture/cut_drr/all/test.txt", help="eval images and labels txt file")
    parser.add_argument('--class_indict_file', type=str, default="./class_indices.json", help="The class name and label id dict")
    parser.add_argument("--weights_category", type=str, default="l", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/spine_fracture/drr/all/l/val_best_model.pth", help="infer weight path")
    opt = parser.parse_args()
    main(opt)
