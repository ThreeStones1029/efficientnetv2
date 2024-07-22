import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import random


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes(root_dir)
        self.image_paths = [os.path.join(root_dir, cls, img) for cls in self.classes for img in os.listdir(os.path.join(root_dir, cls)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = os.path.basename(os.path.dirname(img_path))  # 类别标签
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path, label

# 定义数据集路径和输出路径
data_dir = 'dataset/LA_preoperative_xray_fracture_cut_complete'
output_dir = 'dataset/LA_preoperative_xray_fracture_cut_complete_augmented'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 定义数据增强变换
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.RandomRotation(degrees=15),  # 旋转一定角度
])

def random_crop(image, min_scale=0.9, max_scale=1.0):
    """对图像进行随机裁剪，保留一定比例"""
    width, height = image.size
    crop_scale = random.uniform(min_scale, max_scale)
    crop_width, crop_height = int(width * crop_scale), int(height * crop_scale)
    
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    
    cropped_image = image.crop((left, top, left + crop_width, top + crop_height))
    
    return cropped_image

# 加载数据集
dataset = CustomDataset(root_dir=data_dir, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 设置每张原始图像生成多少增强图像
num_augmentations = 5

# 迭代数据集并保存原始和增强后的图像
for images, img_paths, labels in dataloader:
    image = images.squeeze(0)  # 从batch中移除
    img_path = img_paths[0]
    label = labels[0]
    img_name = os.path.basename(img_path).split('.')[0]

    # 确保输出目录中有对应的类别文件夹
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # 保存原始图像
    original_img_path = os.path.join(label_dir, f"original_{img_name}.png")
    save_image(image, original_img_path)

    # 生成并保存增强后的图像
    for aug_idx in range(num_augmentations):
        augmented_img = data_transforms(image)
        augmented_img_pil = transforms.ToPILImage()(augmented_img)
        augmented_img_cropped = random_crop(augmented_img_pil)
        augmented_img_tensor = transforms.ToTensor()(augmented_img_cropped)
        
        # 保存增强后的图像
        augmented_img_path = os.path.join(label_dir, f"augmented_{img_name}_{aug_idx}.png")
        save_image(augmented_img_tensor, augmented_img_path)

print(f"原始和增强后的图像已保存到 {output_dir}")
