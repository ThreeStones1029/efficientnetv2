'''
Description: 对骨折数据做数据增强
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-07-20 13:49:54
LastEditors: ShuaiLei
LastEditTime: 2024-07-20 14:04:19
'''
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 定义数据增强变换
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.RandomRotation(degrees=15),  # 旋转一定角度
    transforms.RandomCrop(size=(224, 224)),  # 轻微裁剪，裁剪到224x224
    transforms.ToTensor()
])

# 定义数据集路径
data_dir = '/path/to/your/dataset'
output_dir = '/path/to/save/augmented/images'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 加载数据集
dataset = ImageFolder(root=data_dir)

# 定义加载器
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 设置每张原始图像生成多少增强图像
num_augmentations = 5

# 迭代数据集并保存原始和增强后的图像
for idx, (image, label) in enumerate(dataloader):
    image = image.squeeze(0)  # 从batch中移除
    img_name = dataset.imgs[idx][0].split('/')[-1].split('.')[0]

    # 保存原始图像
    original_img_path = os.path.join(output_dir, f"original_{img_name}.png")
    save_image(image, original_img_path)

    # 生成并保存增强后的图像
    for aug_idx in range(num_augmentations):
        augmented_img = data_transforms(image)
        
        # 保存增强后的图像
        augmented_img_path = os.path.join(output_dir, f"augmented_{img_name}_{aug_idx}.png")
        save_image(augmented_img, augmented_img_path)

print(f"原始和增强后的图像已保存到 {output_dir}")
