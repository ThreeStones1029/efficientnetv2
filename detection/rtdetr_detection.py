'''
Description: this file will be used to detect vertebrae in xray.we can use yolov5 or rtdetr_paddle or rtdetr_pytorch
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-17 12:41:00
LastEditors: ShuaiLei
LastEditTime: 2024-07-20 09:35:56
'''
import subprocess
import multiprocessing

def rtdetr_paddle_infer(detection_parameter, infer_dir, output_dir):
    script_parameter = [detection_parameter["envs_path"],
                        detection_parameter["detection_script_path"],
                        "-c", detection_parameter["config_path"],
                        "-o", "weights="+detection_parameter["model_path"],
                        "--infer_dir", infer_dir,
                        "--output_dir",output_dir,
                        "--draw_threshold", "0.5",
                        "--visualize", "True",
                        "--save_results", "True"]
    detection_command = " ".join(script_parameter)
    subprocess.run(detection_command, shell=True)


def rtdetr_pytorch_infer(detection_parameter, infer_dir, output_dir):
    script_parameter = [detection_parameter["envs_path"],
                        detection_parameter["detection_script_path"],
                        "-c", detection_parameter["config_path"],
                        "--resume", detection_parameter["model_path"],
                        "--infer_dir", infer_dir,
                        "--output_dir",output_dir,
                        "--draw_threshold", "0.5",
                        "--visualize", "False",
                        "--save_results", "True"]
    detection_command = " ".join(script_parameter)
    subprocess.run(detection_command, shell=True)