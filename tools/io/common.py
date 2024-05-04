'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-11 08:09:16
LastEditors: ShuaiLei
LastEditTime: 2024-05-04 10:11:22
'''
import os
import json


def load_json_file(json_path):
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(data, json_path):
    dirname = os.path.dirname(json_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print(json_path, "save successfully")  


def get_sub_folder_paths(root_folder):
    """
    this function will be used to get sub_folder_path.
    """
    sub_folder_paths = []
    sub_folder_names = os.listdir(root_folder)
    for sub_folder_name in sub_folder_names:
        if os.path.isdir(os.path.join(root_folder, sub_folder_name)):
            sub_folder_paths.append(os.path.join(root_folder, sub_folder_name))
    return sub_folder_paths 


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path

def join(*args):
    return os.path.join(*args)