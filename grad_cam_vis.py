'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-07-04 13:53:58
LastEditors: ShuaiLei
LastEditTime: 2024-07-04 15:25:40
'''
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from grad_cam.utils import GradCAM, show_cam_on_image, center_crop_img
from model import efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
import argparse


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.weights_category == "s":
        model = efficientnetv2_s(num_classes=args.num_classes)
    if args.weights_category == "m":
        model = efficientnetv2_m(num_classes=args.num_classes)
    if args.weights_category == "l":
        model = efficientnetv2_l(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(model)
    target_layers = [model.head[0]]

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = args.weights_category
    data_transform = transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                         transforms.CenterCrop(img_size[num_model][1]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = args.img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 1  # tabby, tabby cat
    print(input_tensor.shape)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    # Resize grayscale_cam to match the original image size
    # grayscale_cam_resized = cv2.resize(grayscale_cam, (original_img_np.shape[1], original_img_np.shape[0]))

    # Resize grayscale_cam to match the original image size using PIL
    img = np.array(img, dtype=np.float32) / 255.0
    grayscale_cam_pil = Image.fromarray(np.uint8(255 * grayscale_cam))
    grayscale_cam_resized_pil = grayscale_cam_pil.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    grayscale_cam_resized = np.array(grayscale_cam_resized_pil) / 255.0

    visualization = show_cam_on_image(img,
                                      grayscale_cam_resized,
                                      use_rgb=True)
    
    plt.imshow(visualization)
    plt.savefig(args.output_dir)
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 测试数据集所在根目录
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--img_path', type=str, default="dataset/spine_fracture/LA_preoperative_xray_fracture_cut/normal_images/0417-F-069Y1_3.png")
    # Classification paramater
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument("--weights_category", type=str, default="s", help="the pretrain weights category, only s or m or l")
    parser.add_argument('--model_path', type=str, default="weights/spine_fracture/LA_preoperative_xray_fracture_cut/s/val_best_model.pth", help="infer weight path")
    parser.add_argument('--output_dir', type=str, default="normal_grad_cam.png", help="infer image save path")
    opt = parser.parse_args()
    main(opt)
