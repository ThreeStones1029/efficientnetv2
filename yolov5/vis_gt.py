'''
Description: Used to visualize coco format gt annotations in images..
version: 1.0
Author: ThreeStones1029 221620010039@hhu.edu.cn
Date: 2024-02-27 10:16:27
LastEditors: ShuaiLei
LastEditTime: 2024-06-14 03:17:41
'''
from pycocotools.coco import COCO
from glob import glob
import os
import multiprocessing
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict


class GtCOCO(COCO):
    def __init__(self, annotation_file):
        super(GtCOCO, self).__init__(annotation_file)

    # rewrite createIndex
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs, img_idToFilename, FilenameToimg_id, catidTocatname, catnameTocatid = {}, {}, {}, {}, {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
                img_idToFilename[img['id']] = img["file_name"]
                FilenameToimg_id[img["file_name"]] = img["id"]

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
                catidTocatname[cat["id"]] = cat["name"]
                catnameTocatid[cat["name"]] = cat["id"]

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.img_idToFilename = img_idToFilename
        self.FilenameToimg_id = FilenameToimg_id
        self.catidTocatname = catidTocatname
        self.catnameTocatid = catnameTocatid
        self.imgs = imgs
        self.cats = cats


class VisCoCo(GtCOCO):
    def __init__(self, annotation_file, images_folder, bbox_vis_folder=None, rotate_bbox_vis_folder=None):
        super(VisCoCo, self).__init__(annotation_file)
        assert images_folder is not None , "{} is None".format(images_folder)
        self.images_folder = images_folder
        assert bbox_vis_folder is not None , "{} is None".format(bbox_vis_folder)
        if bbox_vis_folder:
            os.makedirs(bbox_vis_folder, exist_ok=True)
        if rotate_bbox_vis_folder:
            os.makedirs(rotate_bbox_vis_folder, exist_ok=True)
        self.bbox_vis_folder = bbox_vis_folder
        self.rotate_bbox_vis_folder = rotate_bbox_vis_folder
        self.draw_text = True
        if self.draw_text:
            try:
                self.font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=20)
            except IOError:
                self.font = ImageFont.load_default(size=20)


    def visualize_bboxes_in_images(self):
        """
        多张图片可视化水平框
        """
        files_path = self.get_files_path()
        # multiprocessing.Pool(8) # 创建8个进程，提高代码处理效率
        # with multiprocessing.Pool(8) as pool:
        #     list(tqdm(pool.imap(self.visualize_bboxes_in_image, [(file_path) for file_path in sorted(files_path) if os.path.basename(file_path) in self.FilenameToimg_id.keys()]), total=len(files_path), desc="vis bbox"))
        for file_path in sorted(files_path):
            self.visualize_bboxes_in_image(file_path)

    def visualize_bboxes_in_image(self, file_path):
        """
        单张图片可视化水平框
        """
        file_name = os.path.basename(file_path)
        if file_name in self.FilenameToimg_id.keys():
            image_id = self.FilenameToimg_id[file_name]
            save_image_path = os.path.join(self.bbox_vis_folder, file_name)
            image_info = self.loadImgs(image_id)[0]
            file_name = image_info['file_name']
            img_path = os.path.join(self.images_folder, file_name)
            image = Image.open(img_path).convert('RGB')
            image = ImageOps.exif_transpose(image)
            # 获取这张图片的ann
            ann_ids = self.getAnnIds(imgIds=image_id)
            annotations = self.loadAnns(ann_ids)
            # 可视化
            image = self.draw_bbox(image, annotations)
            # 保存
            self.save_result(save_image_path, image)


    def draw_bbox(self, image, annotations):
        """
        Draw bbox on image 分别可视化bbox和label是为了文字不被挡住
        """
        draw = ImageDraw.Draw(image)
        for ann in annotations:
            bbox = ann['bbox']
            # draw bbox
            if len(bbox) == 4:
                # draw bbox
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                draw.line(
                    [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                    (xmin, ymin)],
                    width=2,
                    fill='red')
            else:
                print('the shape of bbox must be [M, 4]')

        for ann in annotations:
            catid, bbox = ann['category_id'], ann['bbox']
            xmin, ymin, w, h = bbox
            # draw label
            if self.catidTocatname:
                text = "{} ".format(self.catidTocatname[catid])
            else:
                catname = ann['category_name']
                text = "{}".format(catname)
            # tw, th = draw.textsize(text)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=self.font)
            tw, th = right - left, bottom - top
            #label框
            draw.rectangle([(xmin + 1, ymin + 1), (xmin + tw + 1, ymin + th + 1 + 10)], fill='white') 
            # draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill = color)
            # label文字 
            # (xmin + 1, ymin - th)
            draw.text((xmin + 1, ymin + 1), text, fill='red', font=self.font) 
            # draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
        return image
    

    def visualize_rotate_bboxes_in_images(self):
        """
        多张图片可视化旋转框
        """
        files_path = self.get_files_path()
        # multiprocessing.Pool(8) # 创建8个进程，提高代码处理效率
        with multiprocessing.Pool(8) as pool:
            list(tqdm(pool.imap(self.visualize_rotate_bboxes_in_image, [(file_path) for file_path in sorted(files_path) if os.path.basename(file_path) in self.FilenameToimg_id.keys()]), total=len(files_path), desc="vis rotation bbox"))


    def visualize_rotate_bboxes_in_image(self, file_path):
        """
        单张图片可视化旋转框
        """
        file_name = os.path.basename(file_path)
        image_id = self.FilenameToimg_id[file_name]
        save_image_path = os.path.join(self.rotate_bbox_vis_folder, file_name)
        image_info = self.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        img_path = os.path.join(self.images_folder, file_name)
        image = Image.open(img_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        # 获取这张图片的ann
        ann_ids = self.getAnnIds(imgIds=image_id)
        annotations = self.loadAnns(ann_ids)
        # 可视化
        image = self.draw_rotate_bbox(image, annotations)
        # 保存
        self.save_result(save_image_path, image)


    def draw_rotate_bbox(self, image, annotations):
        """
        Draw bbox on image 分别可视化bbox和label是为了文字不被挡住
        """
        draw = ImageDraw.Draw(image)
        for ann in annotations:
            rotate_bbox = ann['segmentation']
            # draw rotate_bbox
            if len(rotate_bbox[0]) == 8:
                # draw bbox
                x1, y1 = rotate_bbox[0][0], rotate_bbox[0][1]
                x2, y2 = rotate_bbox[0][2], rotate_bbox[0][3]
                x3, y3 = rotate_bbox[0][4], rotate_bbox[0][5]
                x4, y4 = rotate_bbox[0][6], rotate_bbox[0][7]
                draw.line([(x1, y1), (x2, y2), (x3, y3), (x4, y4),(x1, y1)], width=2, fill='red')
            else:
                print('the shape of rotation bbox shape must be [1, 8]')

        for ann in annotations:
            catid, rotate_bbox = ann['category_id'], ann['segmentation']
            # rect_points = np.array([[rotate_bbox[0][0], rotate_bbox[0][1]],
            #                         [rotate_bbox[0][2], rotate_bbox[0][3]],
            #                         [rotate_bbox[0][4], rotate_bbox[0][5]],
            #                         [rotate_bbox[0][6], rotate_bbox[0][7]]])
            # (xmin, ymin) = np.min(rect_points, axis=0)
            xmin, ymin = rotate_bbox[0][0], rotate_bbox[0][1]
            # draw label
            if self.catidTocatname:
                text = "{} ".format(self.catidTocatname[catid])
            else:
                catname = ann['category_name']
                text = "{}".format(catname)
            # tw, th = draw.textsize(text)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=self.font)
            tw, th = right - left, bottom - top
            #label框
            draw.rectangle([(xmin + 1, ymin + 1), (xmin + tw + 1, ymin + th + 1 + 10)], fill='white') 
            # draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill = color)
            # label文字 
            # (xmin + 1, ymin - th)
            draw.text((xmin + 1, ymin + 1), text, fill='red',font=self.font) 
            # draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
        return image
    

    def save_result(self, save_path, image):
        """
        save visual result 
        """
        image.save(save_path, quality=95)     
        # print("coco bbox visual results save in {}".format(save_path))


    def get_files_path(self):
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        files_path = set()
        for ext in exts:
            files_path.update(glob('{}/*.{}'.format(self.images_folder, ext)))
        files_path = list(files_path)
        assert len(files_path) > 0, "no image found in {}".format(files_path)
        return files_path


if __name__ == "__main__":
    vis = VisCoCo(annotation_file="/home/ABLSpineLevelCheck_single/datasets/coco128/annotations/train2017.json",
            images_folder="/home/ABLSpineLevelCheck_single/datasets/coco128/images/train2017",
            bbox_vis_folder="/home/ABLSpineLevelCheck_single/datasets/coco128/gt")
    vis.visualize_bboxes_in_images()
