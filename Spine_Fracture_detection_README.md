<!--
 * @Description: 
 * @version: 
 * @Author: ThreeStones1029 2320218115@qq.com
 * @Date: 2024-04-02 14:01:37
 * @LastEditors: ShuaiLei
 * @LastEditTime: 2024-04-09 09:08:08
-->
# 说明
本文将用于记录骨折检测,总体流程将会是先检测,再对检测框里面的椎体进行二分类判断是否骨折

# 数据集制作(代码在drr_utils仓库中)
## DRR数据集制作
* 由于骨折不明显时,正位较难判别椎体是否骨折,所以目前只做侧位的骨折检测
* 根据verse2019的骨折分级标注,选出在T9-L6存在三级骨折的CT33例,然后手动挑选了17例较好的
* 再加上本地数据集22例,总共39例数据生成DRR
~~~bash

~~~
* 再根据生成DRR的质量筛选出17例CT,其中有7例通过裁剪全脊柱得到
~~~bash

~~~
* 检测标注与骨折标注转换
~~~bash

~~~
* 挑选出骨折与正常椎体,比例为1:1
~~~bash

~~~
## 真实X线片数据集制作
~~~bash

~~~
## 数据集划分
~~~bash

~~~
# DRR and Intraoperative X-ray spine fracture location and vertebrae detection.

## DRR
DRR train predict eval and vis process can find details in [here](document/DRR.md)

## Intraoperative X-ray 
Intraoperative X-ray train predict eval and vis process can find details in [here](document/Intraoperative_X-ray.md)
