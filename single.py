# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 上午10:33
@file: single.py
@author: zj
@description: 
"""

import yaml
import argparse

import torch

from models.yolov3 import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # 创建YOLOv3
    model = YOLOv3(cfg['MODEL'])
    model.eval()
    print(model)

    import random
    import numpy as np

    seed = 1  # seed必须是int，可以自行设置
    random.seed(seed)
    np.random.seed(seed)  # numpy产生的随机数一致
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
        torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
        # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
        # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
        torch.backends.cudnn.deterministic = True

        # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
        torch.backends.cudnn.benchmark = False

    # data = torch.randn((1, 3, 416, 416))
    # data = torch.randn((1, 3, 608, 608))
    data = torch.randn((1, 3, 640, 640))
    print(data.reshape(-1)[:20])
    outputs = model(data)
    print(outputs.shape)

    print(outputs.reshape(-1)[:20])
