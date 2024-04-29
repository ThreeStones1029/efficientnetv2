<!--
 * @Description: 
 * @version: 
 * @Author: ThreeStones1029 2320218115@qq.com
 * @Date: 2024-04-29 07:59:51
 * @LastEditors: ShuaiLei
 * @LastEditTime: 2024-04-29 12:21:01
-->
# Download verse2020 dataset
[VerSe 2020 (subject based data structure)](https://osf.io/4skx2/)
# Merge Train val and test dataset to one folder

# Merge_raw_data_and_derivates
~~~bash
python tools/data/verse2020_dataset_preprocess.py
~~~

## delete the Redundant json files, one sub_folder only save one json file.
check the sub_folder and delete the Redundant json file.
~~~bash
python tools/data/verse2020_dataset_preprocess.py
~~~

