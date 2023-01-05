# -*- coding: utf-8 -*-

"""
@date: 2023/1/5 上午11:29
@file: dataset.py
@author: zj
@description: 
"""

import yaml

from dataset.cocodataset import COCODataset

if __name__ == '__main__':
    cfg_file = 'config/yolov3_default.cfg'
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    imgsize = cfg['TRAIN']['IMGSIZE']
    dataset = COCODataset(
        # 模型类型，针对YOLO，会转换label格式
        model_type=cfg['MODEL']['TYPE'],
        # 数据集根路径，加载标注文件以及图像文件
        data_dir='COCO/',
        # 输入图像大小
        img_size=imgsize,
        # 数据增强
        augmentation=cfg['AUGMENTATION'],
    )

    print(dataset)

    res = dataset.__getitem__(332)
