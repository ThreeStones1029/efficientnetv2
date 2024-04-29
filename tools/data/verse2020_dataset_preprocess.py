'''
Description: 
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-29 07:56:54
LastEditors: ShuaiLei
LastEditTime: 2024-04-29 12:55:34
'''
import os
import sys
sys.path.insert(0, "/home/efficientnetV2")
import shutil
from tqdm import tqdm
from tools.io.common import load_json_file


def check_sub_folders(verse2020_root):
    """
    This function will be used to check the verse2020 dataset.verse2020 dataset ct nums=214.
    param: verse2020_root: the train val and test folder.
    """
    for sub_folder_name in tqdm(os.listdir(verse2020_root), desc="check subfolder"):
        sub_folder_path = os.path.join(verse2020_root, sub_folder_name)
        for file in os.listdir(sub_folder_path):
            if file.endswith(".json"):
                data = load_json_file(os.path.join(sub_folder_path, file))
                if isinstance(data, dict):
                    os.remove(os.path.join(sub_folder_path, file))
                    print(os.path.join(sub_folder_path, file), "deleted")
                if isinstance(data, list) and "direction" not in data[0]:
                    os.remove(os.path.join(sub_folder_path, file))
                    print(os.path.join(sub_folder_path, file), "deleted")


def merge_raw_data_and_derivatives(verse2020_root):
    """
    The function will  be used to merge verse2020.
    param: verse2020_root: the train val and test folder.
    """
    raw_data_folder = os.path.join(verse2020_root, "rawdata")
    derivatives_folder = os.path.join(verse2020_root, "derivatives")
    for sub_folder_name in tqdm(os.listdir(raw_data_folder), desc="merge subfolder"):
        raw_data_sub_folder_path = os.path.join(raw_data_folder, sub_folder_name)
        derivatives_sub_folder_path = os.path.join(derivatives_folder, sub_folder_name)
        merge_sub_folder_path = os.path.join(verse2020_root, sub_folder_name)
        os.makedirs(merge_sub_folder_path, exist_ok=True)
        for file in os.listdir(raw_data_sub_folder_path):
            shutil.copy(os.path.join(raw_data_sub_folder_path, file), os.path.join(merge_sub_folder_path, file))
        for file in os.listdir(derivatives_sub_folder_path):
            shutil.copy(os.path.join(derivatives_sub_folder_path, file), os.path.join(merge_sub_folder_path, file))
    print("merge successfully!")


def rename_verse2020_subfolder_files(verse2020_root):
    """
    the function will be used to rename the verse2020_dataset subfolder.
    before rename example:
       sub-gl003_dir-ax_ct.nii.gz  
       sub-gl003_dir-ax_seg-subreg_ctd.json  
       sub-gl003_dir-ax_seg-vert_msk.nii.gz  
       sub-gl003_dir-ax_seg-vert_snp.png 
    after rename example:
       sub-gl003.nii.gz  
       sub-gl003.json  
       sub-gl003.nii.gz  
       sub-gl003.png 
    param: verse2020_root: the train val and test folder.
    """
    for sub_folder_name in tqdm(os.listdir(verse2020_root), desc="rename subfiles"):
        sub_folder_path = os.path.join(verse2020_root, sub_folder_name)
        for file in os.listdir(sub_folder_path):
            if file.endswith(".json"):
                os.rename(os.path.join(sub_folder_path, file), os.path.join(sub_folder_path, sub_folder_name + ".json"))
            if file.endswith(".png"):
                os.rename(os.path.join(sub_folder_path, file), os.path.join(sub_folder_path, sub_folder_name + ".png"))
            if file.endswith("ax_ct.nii.gz"):
                os.rename(os.path.join(sub_folder_path, file), os.path.join(sub_folder_path, sub_folder_name + ".nii.gz"))
            if file.endswith("seg-vert_msk.nii.gz"):
                os.rename(os.path.join(sub_folder_path, file), os.path.join(sub_folder_path, sub_folder_name + "_seg.nii.gz"))
    print("rename successfully!")
            

if __name__ == "__main__":
    # # Merge_raw_data_and_derivates
    # merge_raw_data_and_derivatives("/root/share/ShuaiLei/verse2020_dataset")

    # # check subfolder
    # check_sub_folders("/root/share/ShuaiLei/verse2020_dataset")

    # rename subfolder subfiles
    rename_verse2020_subfolder_files("/root/share/ShuaiLei/verse2020_dataset")