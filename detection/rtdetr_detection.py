'''
Description: The function is rtdetr detection
version: 
Author: ThreeStones1029 2320218115@qq.com
Date: 2024-04-09 09:19:26
LastEditors: ShuaiLei
LastEditTime: 2024-04-16 02:10:56
'''
def rtdetr_infer():
    """
    """
    script_parameter = [object_detection_parameter["envs_path"],
                        object_detection_parameter["detection_script_path"],
                        "-c", object_detection_parameter["config_path"],
                        "--infer_dir", images_path,
                        "--output_dir", detection_result_save_path,
                        "--draw_threshold", "0.6",
                        "--use_vdl", "False",
                        "--save_results", "True"]
    detection_command = " ".join(script_parameter)
    subprocess.run(detection_command, shell=True)