<!--
 * @Description: 
 * @version: 
 * @Author: ThreeStones1029 2320218115@qq.com
 * @Date: 2024-04-09 09:02:50
 * @LastEditors: ShuaiLei
 * @LastEditTime: 2024-04-14 08:52:27
-->
[简体中文](DRR_CN.md) | English
# Dataset
There are three kinds of datasets which LA, AP, and all. 
These experiments were done to verify the effect of the positive lateral position.
## DRR
The final AP and LA drr images number table as follow.
| No | AP | LA | all |
|:---:|:---:|:---:|:---:|
| fracture_images | 424 | 410 | 834 |
| normal_images | 444 | 390 | 834 |

## SpineXR 

# Train

# Predict

# Val

# Result
## DRR vertebrae detection
### dataset split
| No | train | val | test |
|:---:|:---:|:---:|:---:|
| number | 468 | 156 | 156 |

### train parameter
| parameter | value | parameter | value |
|:---:|:---:|:---:|:---:|
| epoch | 200 | lr | 0.00025 |


## DRR Classification
| No | train(number) | val(number) | s(77.84MB)accuracy | m(203.15MB)accuracy | l(449.72MB)accuracy | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| AP | 696 | 172 | 91.279% | 91.86% | 93.6% |
| LA | 640 | 160 |87.5% | 89.375% | 94.375% |
| all | 1336 | 332 | 87.65% | 91.867% | 92.168% |
