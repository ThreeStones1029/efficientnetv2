'''
Description:
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-09 09:20:08
LastEditors: ShuaiLei
LastEditTime: 2024-04-17 12:52:50
'''
import subprocess


def yolov5_infer(envs_path, detection_script_path, config_path, infer_dir, infer_output_dir):
    script_parameter = [envs_path,
                        detection_script_path,
                        "-c", config_path,
                        "--infer_dir", infer_dir,
                        "--output_dir",infer_output_dir,
                        "--draw_threshold", "0.5",
                        "--save_results", "True"]
    detection_command = " ".join(script_parameter)
    subprocess.run(detection_command, shell=True)