import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    parser.add_argument('--ckpt', type=str,
                        help='path to the check point file')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='path to weights file')
    parser.add_argument('--image', type=str)
    parser.add_argument('--background', action='store_true',
                        default=False, help='background(no-display mode. save "./output.png")')
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')
    args = parser.parse_args()
    return args


def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    args = parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    # 输入图像大小
    imgsize = cfg['TEST']['IMGSIZE']
    # 创建YOLOv3
    model = YOLOv3(cfg['MODEL'])

    confthre = cfg['TEST']['CONFTHRE']
    nmsthre = cfg['TEST']['NMSTHRE']

    if args.detect_thresh:
        confthre = args.detect_thresh

    # BGR
    img = cv2.imread(args.image)
    # [H, W, C] -> [C, H, W]　同时　BGR -> RGB
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    # 图像归一化 + 通道转换（[H, W, C] -> [C, H, W]）
    img = np.transpose(img / 255., (2, 0, 1))
    # [C, H, W] -> [1, C, H, W]
    img = torch.from_numpy(img).float().unsqueeze(0)

    if args.gpu >= 0:
        model.cuda(args.gpu)
        img = Variable(img.type(torch.cuda.FloatTensor))
    else:
        img = Variable(img.type(torch.FloatTensor))

    assert args.weights_path or args.ckpt, 'One of --weights_path and --ckpt must be specified'

    if args.weights_path:
        print("loading yolo weights %s" % (args.weights_path))
        # 加载YOLO权重
        parse_yolo_weights(model, args.weights_path)
    elif args.ckpt:
        print("loading checkpoint %s" % (args.ckpt))
        state = torch.load(args.ckpt)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    model.eval()

    with torch.no_grad():
        # img: [1, 3, 416, 416]
        outputs = model(img)
        # outputs: [B, N_bbox, 4(xywh)+1(conf)+num_classes]
        outputs = postprocess(outputs, 80, confthre, nmsthre)

    if outputs[0] is None:
        print("No Objects Deteted!!")
        return

    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    bboxes = list()
    classes = list()
    colors = list()

    # x1/y1: 左上角坐标
    # x2/y2: 右下角坐标
    # conf: 置信度
    # cls_conf: 分类置信度
    # cls_pred: 分类下标
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
        cls_id = coco_class_ids[int(cls_pred)]
        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' %
              (coco_class_names[cls_id], cls_conf.item()))
        box = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append(box)
        classes.append(cls_id)
        colors.append(coco_class_colors[int(cls_pred)])

    if args.background:
        import matplotlib
        matplotlib.use('Agg')

    from utils.vis_bbox import vis_bbox
    import matplotlib.pyplot as plt

    vis_bbox(
        img_raw, bboxes, label=classes, label_names=coco_class_names,
        instance_colors=colors, linewidth=2)
    plt.show()

    if args.background:
        plt.savefig('output.png')


if __name__ == '__main__':
    main()
