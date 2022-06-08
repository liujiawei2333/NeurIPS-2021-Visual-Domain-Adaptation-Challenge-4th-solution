# NeurIPS 2021 Visual Domain Adaptation Challenge 4th solution

This is our code for Visual Domain Adaptation Challenge 2021(visDA-2021).

## Requirements:

`pytorch,torchvision,tensorboardX,apex,sklearn`

## Pre-training：

`python train_pre.py --source_data path1 --target_data path2 --network adveffi-b4 --save_path path3  `

`source_data`:path of the source data

`target_data`:path of the target data

`network`:adveffi-b4/adveffi-b5/adveffi-b6

`save_path`:path of model and forecast files

## stage 1 training

`python train_pseudo.py --source_data path1 --target_data path2 --network adveffi-b4 --save_path path3 --pseudo_times 1`

`source_data`:path of the source data

`target_data`:pseudo label file for target data

`network`:adveffi-b4/adveffi-b5/adveffi-b6

`save_path`:path of model and forecast files

`pseudo_times`:pseudo label training rounds

## stage 2 training

`python train_pseudo.py --source_data path1 --target_data path2 --network adveffi-b4 --save_path path3 --pseudo_times 1`

`source_data`:path of the source data

`target_data`:pseudo label file for test data

`network`:adveffi-b4/adveffi-b5/adveffi-b6

`save_path`:path of model and forecast files

`pseudo_times`:pseudo label training rounds

Note: In stage 1 training, line 304 of 'train_pseudo-py' uses test; In stage 2 training, line 304 of 'train_pseudo-py' uses test2.

## test

`python eval_test.py --target_data path1 --pseudo_data path1 --network adveffi-b4 --save_path path3 --pseudo_times 1`

`target_data`和`pseudo_data`:path of the test data

`network`:adveffi-b4/adveffi-b5/adveffi-b6

`save_path`:path of model and forecast files

`pseudo_times`:pseudo label training rounds

## model ensemble

1. Get the prediction files for the three models and rename them.

|               File                |   new name    |
| :-------------------------------: | :-----------: |
| adapt_pred.txt of EfficientNet-B6 | adapt_pred_B6 |
| adapt_pred.txt of EfficientNet-B5 | adapt_pred_B5 |
| adapt_pred.txt of EfficientNet-B4 | adapt_pred_B4 |

2. Use `txt2xlsx.py` to process three txt files into xlsx files.

3. Put the three xlsx files in one folder, make sure there are no other hidden files in that folder.

4. Generate the xlsx file after model ensembel with `ensemble_from_txt.py`.

5. Delete the first row (column name) and first column (ordinal column) of the generated xlsx file.

6. Use `xlsx2txt.py` to process the generated xlsx file into txt files.

7. Use `del_txt.py` to delete the redundant absolute paths in the front of the txt file after processing.