from __future__ import division

from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from dataset.cocodataset import *

import os
import argparse
import yaml
import random

import torch
from torch.autograd import Variable
import torch.optim as optim

"""
操作流程：

1. 解析命令行参数 + 配置文件
2. 训练配置：
    1. 初始化模型，加载预训练权重
    2. 初始化数据类、批量加载器、数据集评估器
    3. 初始化优化器 + 学习率调度器
3. 训练：
    1. 每轮迭代包含subdivision * batchsize个图像
    2. 每轮计算后，执行梯度更新以及学习率更新
    3. 每隔10轮计算，打印使用学习率、平均损失以及图像缩放大小
    4. 每隔10轮重新设置图像缩放大小（32的倍数，取值范围在[320, 608]）
4. 评估：
    1. 每隔4000轮评估一次
"""

def parse_args():
    parser = argparse.ArgumentParser()
    # 模型配置以及训练配置
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg',
                        help='config file. see readme')
    # 权重路径
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    # 数据加载线程数
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of workers')
    # 间隔多少次训练保存权重
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    # 间隔多少次进行评估
    parser.add_argument('--eval_interval', type=int,
                        default=4000, help='interval between evaluations')
    # ???
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    # 权重保存根路径
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    # 是否使用GPU（注意：本仓库仅实现了单GPU训练）
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard', help='tensorboard path for logging', type=str, default=None)
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        # cfg = yaml.load(f)
        cfg = yaml.safe_load(f)

    print("successfully loaded config file: ", cfg)

    # 动量
    momentum = cfg['TRAIN']['MOMENTUM']
    # 衰减
    decay = cfg['TRAIN']['DECAY']
    # warmup迭代次数
    burn_in = cfg['TRAIN']['BURN_IN']
    # 最大迭代次数
    iter_size = cfg['TRAIN']['MAXITER']
    # 权重衰减阶段
    steps = eval(cfg['TRAIN']['STEPS'])
    # 单次训练批量大小
    batch_size = cfg['TRAIN']['BATCHSIZE']
    # 子批次，累加subdivision次训练后反向传播梯度
    subdivision = cfg['TRAIN']['SUBDIVISION']
    # 阈值
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    # 缩放大小
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    # 初始学习率
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision

    # 关于批量大小，是因为目标检测训练中，从单张图片中采集了大量的真值框参与训练
    # 所以单次训练的数据量很小，为了更稳定的进行梯度更新，需要累加多次批量运算的梯度（也就是梯度累加）
    # 所以有效批量大小 = batch_size(单次批量大小) * iter_size(累计批次)
    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Learning rate setup
    def burnin_schedule(i):
        if i < burn_in:
            # 在warmup阶段，使用线性学习率进行递增
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # Initiate model
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    # 预训练权重加载，共两种方式
    if args.weights_path:
        # 方式一：Darknet格式权重文件
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    elif args.checkpoint:
        # 方式二：Pytorch格式权重文件
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        # GPU训练
        print("using cuda")
        model = model.cuda()

    if args.tfboard:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard)

    # 训练模式
    model.train()

    imgsize = cfg['TRAIN']['IMGSIZE']
    # COCO数据集
    dataset = COCODataset(
        # 模型类型，针对YOLO，会转换label格式
        model_type=cfg['MODEL']['TYPE'],
        # 数据集根路径，加载标注文件以及图像文件
        data_dir='COCO/',
        # 输入图像大小
        img_size=imgsize,
        # 数据增强
        augmentation=cfg['AUGMENTATION'],
        debug=args.debug
    )

    # 数据加载器，每次加载batch_size个图像数据以及对应标签
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
    dataiterator = iter(dataloader)

    # COCO评估器
    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                                 data_dir='COCO/',
                                 img_size=cfg['TEST']['IMGSIZE'],
                                 confthre=cfg['TEST']['CONFTHRE'],
                                 nmsthre=cfg['TEST']['NMSTHRE'])

    # 指定输入模型数据的类型
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # optimizer setup
    # set weight decay only on conv.weight
    # 仅针对卷积层权重执行权重衰减
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{'params': value, 'weight_decay': decay * batch_size * subdivision}]
        else:
            params += [{'params': value, 'weight_decay': 0.0}]
    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size * subdivision)

    # 初始化迭代次数
    iter_state = 0

    # Resume
    if args.checkpoint:
        if 'optimizer_state_dict' in state.keys():
            optimizer.load_state_dict(state['optimizer_state_dict'])
            iter_state = state['iter'] + 1

    # 学习率调度器
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # start training loop
    # 训练，共执行iter_size次（50w）
    for iter_i in range(iter_state, iter_size + 1):
        # 下面的model均保持cuda类型（如果使用GPU训练的话）

        # COCO evaluation
        if iter_i % args.eval_interval == 0 and iter_i > 0:
            # 每隔eval_interval进行评估
            print("Begin evaluating ...")
            ap50_95, ap50 = evaluator.evaluate(model)
            model.train()
            if args.tfboard:
                tblogger.add_scalar('val/COCOAP50', ap50, iter_i)
                tblogger.add_scalar('val/COCOAP50_95', ap50_95, iter_i)

        # subdivision loop
        # 子批次循环，进行梯度累加
        # 也就是说每次迭代（iter_i）的总批量大小为batch_size * subdivision
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            # 转换到指定数据格式dtype
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            # 在训练阶段，model返回整体损失
            loss = model(imgs, targets)
            # 梯度计算
            loss.backward()

        # 训练完batch_size * subdivision后执行梯度更新
        optimizer.step()
        scheduler.step()

        if iter_i % 10 == 0:
            # logging
            # 计算当前学习率
            current_lr = scheduler.get_lr()[0] * batch_size * subdivision
            # 打印当前迭代次数／总迭代次数／当前学习率／各个子损失／当前指定图像大小
            print('[Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, total %f, imgsize %d]'
                  % (iter_i, iter_size, current_lr,
                     model.loss_dict['xy'], model.loss_dict['wh'],
                     model.loss_dict['conf'], model.loss_dict['cls'],
                     model.loss_dict['l2'], imgsize),
                  flush=True)

            if args.tfboard:
                tblogger.add_scalar('train/total_loss', model.loss_dict['l2'], iter_i)

            # random resizing
            # 如果设置了随机缩放，那么每隔10次训练后重新指定输入图像大小
            if random_resize:
                imgsize = (random.randint(0, 9) % 10 + 10) * 32
                dataset.img_shape = (imgsize, imgsize)
                dataset.img_size = imgsize
                # 重新设置数据加载器，因为是打乱模式，可以保证提取数据不会保持一致
                # 因为手动递增迭代次数，所以不需要依赖数据加载器进行计数
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
                dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            torch.save({'iter': iter_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        },
                       os.path.join(args.checkpoint_dir, "snapshot" + str(iter_i) + ".ckpt"))
    if args.tfboard:
        tblogger.close()


if __name__ == '__main__':
    main()
