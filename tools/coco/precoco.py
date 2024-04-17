'''
Description:Since the predicted json file only annotations, this file is intended to mimic coco's official API to implement some of the same functionality. 
version: 
Author: ThreeStones1029 221620010039@hhu.edu.cn
Date: 2023-09-26 15:46:35
LastEditors: ShuaiLei
LastEditTime: 2024-04-17 14:10:54
'''
import time
from collections import defaultdict
from tools.io.common import load_json_file


class PreCOCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.cat_id2cat_name = dict()
        print('loading annotations into memory...')
        tic = time.time()
        # 通过路径下载
        if type(annotation_file) == str:
            dataset = load_json_file(annotation_file)
            assert type(dataset)==list, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = {"annotations": dataset}
        # 直接加载list
        elif type(annotation_file) == list:
            self.dataset = {"annotations": annotation_file}
        else:
            print("annotation_file must be path or list")
        self.createIndex()
        print('Done (t={:0.2f}s)'.format(time.time()- tic))


    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs, img_idToFilename = {}, {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                img_idToFilename[ann['image_id']] = ann["file_name"]
        print('index created!')
        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.img_idToFilename = img_idToFilename
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        if 'info' in self.dataset:
            for key, value in self.dataset['info'].items():
                print('{}: {}'.format(key, value))
        else:
            print("dataset don't have info, please check your json file")


    def gen_cat_id2cat_name(self):
        for ann in self.dataset["annotations"]:
            if ann["category_id"] not in self.cat_id2cat_name.keys():
                self.cat_id2cat_name[ann["category_id"]] = ann["category_name"]
        return self.cat_id2cat_name


if __name__ == '__main__':
    coco_dt = PreCOCO('infer_output/semantic/bbox.json')
    print(coco_dt.imgid2filename)