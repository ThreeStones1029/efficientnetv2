'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-11 08:09:16
LastEditors: ShuaiLei
LastEditTime: 2024-04-11 08:13:17
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