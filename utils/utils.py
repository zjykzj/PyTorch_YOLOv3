from __future__ import division
import torch
import numpy as np
import cv2


def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        # 按照置信度大小进行排序
        order = score.argsort()[::-1]
        bbox = bbox[order]
    # 计算预测框面积
    # bbox[:, 2:] - bbox[:, :2]:
    #   (x2 - x1)/(y2 - y1)
    # bbox_area: [N_bbox]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        # 计算每个预测框与其他预测框的IoU
        # 首先计算交集面积，
        # top-left: [2(top, left)]
        tl = np.maximum(b[:2], bbox[selec, :2])
        # bottom-right: [2(bottom, right)]
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            # 如果iou大于阈值，说明该预测框与前面已确认的预测框高度重叠，需要舍弃
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            # 是否对每一个类别的预测框数目进行约束
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    # 输入prediction: [B, N_bbox, 4+1+80]
    box_corner = prediction.new(prediction.shape)
    # 计算左上角坐标x0 = x_c - w/2
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # 计算左上角坐标y0 = y_c - h/2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # 计算右下角坐标x1 = x_c + w/2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # 计算右下角坐标y1 = y_c + h/2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # 计算每幅图像的预测结果
        # Filter out confidence scores below threshold
        # 计算每个预测框对应的最大类别概率
        # [N_bbox, num_classes] -> ｛分类概率，对应下标｝
        class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # [N_bbox]
        class_pred = class_pred[0]
        # 置信度掩码 [N_bbox] -> [N_bbox]
        # Pr(Class_i | Object) * Pr(Object) = Pr(Class_i)
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre)
        conf_mask = conf_mask.squeeze()
        # 过滤不符合置信度阈值的预测框
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        # 如果所有预测框都已经舍弃，继续下一张图片的预测框计算
        if not image_pred.size(0):
            continue
        # Get detections with higher confidence scores than the threshold
        # (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre)得到一个二维矩阵：[N_bbox, 80]
        # nonzero()得到一个二维矩阵：[N_nonzero, 2]
        # N_nonzero表示二维矩阵[N_bbox, 80]中每一行不为0的数目
        # [N_nonzero, 0]表示行下标
        # [N_nonzero, 1]表示列下标
        ind = (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre).nonzero()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # 获取预测结果
        # image_pred[ind[:, 0], :5]: 每个预测框的预测坐标 + 置信度
        # image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1): 每个预测框的分类概率
        # ind[:, 1].float().unsqueeze(1): 每个预测框的分类下标
        detections = torch.cat((
            image_pred[ind[:, 0], :5],
            image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1),
            ind[:, 1].float().unsqueeze(1)
        ), 1)
        # Iterate through all predicted classes
        # 统计所有预测框对应的类别列表
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # 计算特定类别的预测框
            # Get the detections with the particular class
            # 获取特定类别的预测框列表
            detections_class = detections[detections[:, -1] == c]
            nms_in = detections_class.cpu().numpy()
            # 输入
            # nms_in[:, :4]: 特定类别的预测框坐标
            # nms_thre: NMS阈值
            # nms_in[:, 4] * nms_in[:, 5]:
            #   Pr(Object) * Pr(Class_i | Object) = Pr(Class_i)
            #   属于该类别的置信度
            nms_out_index = nms(
                nms_in[:, :4], nms_thre, score=nms_in[:, 4] * nms_in[:, 5])
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    # bboxes_a: [N_a, 4]
    # bboxes_b: [N_b, 4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        # xyxy: x_top_left, y_top_left, x_bottom_right, y_bottom_right
        # 计算交集矩形的左上角坐标
        # torch.max([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        # torch.max: 双重循环
        #   第一重循环 for i in range(N_a)，遍历boxes_a, 获取边界框i，大小为[2]
        #       第二重循环　for j in range(N_b)，遍历bboxes_b，获取边界框j，大小为[2]
        #           分别比较i[0]/j[0]和i[1]/j[1]，获取得到最大的x/y
        #   遍历完成后，获取得到[N_a, N_b, 2]
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        # 计算交集矩形的右下角坐标
        # torch.min([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # 计算bboxes_a的面积
        # x_bottom_right/y_bottom_right - x_top_left/y_top_left = w/h
        # prod([N, w/h], 1) = [N], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # x_center/y_center -> x_top_left, y_top_left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        # x_center/y_center -> x_bottom_right/y_bottom_right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # prod([N_a, w/h], 1) = [N_a], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # 判断符合条件的结果：x_top_left/y_top_left < x_bottom_right/y_bottom_right
    # [N_a, N_b, 2] < [N_a, N_b, 2] = [N_a, N_b, 2]
    # prod([N_a, N_b, 2], 2) = [N_a, N_b], 数值为1/0
    en = (tl < br).type(tl.type()).prod(dim=2)
    # 首先计算交集w/h: [N_a, N_b, 2] - [N_a, N_b, 2] = [N_a, N_b, 2]
    # 然后计算交集面积：prod([N_a, N_b, 2], 2) = [N_a, N_b]
    # 然后去除不符合条件的交集面积
    # [N_a, N_b] * [N_a, N_b](数值为1/0) = [N_a, N_b]
    # 大小为[N_a, N_b]，表示bboxes_a的每个边界框与bboxes_b的每个边界框之间的IoU
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    # 计算IoU
    # 首先计算所有面积
    # area_a[:, None] + area_b - area_i =
    # [N_a, 1] + [N_b] - [N_a, N_b] = [N_a, N_b]
    # 然后交集面积除以所有面积，计算IoU
    # [N_a, N_b] / [N_a, N_b] = [N_a, N_b]
    return area_i / (area_a[:, None] + area_b - area_i)


def label2yolobox(labels, info_img, maxsize, lrflip):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    # h/w: 原始图像高/宽
    # nh/nw: 缩放图像高/宽（未填充）
    # dx/dy: 填充大小
    h, w, nh, nw, dx, dy = info_img
    # x1/y1/b_w/b_h -> x1/y1/x2/y2
    # 设置成和原始图像大小的比率
    x1 = labels[:, 1] / w
    y1 = labels[:, 2] / h
    x2 = (labels[:, 1] + labels[:, 3]) / w
    y2 = (labels[:, 2] + labels[:, 4]) / h

    # (x1 + x2)/2得到x_center
    # x_center*nw得到缩放后的坐标
    # xc*nw + dx得到填充后的坐标
    # 最后除以目标大小得到对应比率
    labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
    # 对于坐标框的宽/高而言，没有填充值，所以计算其对应结果图像的宽/高比率为
    # b_w / src_w * resized_w / dst_w
    # b_h / src_h * resized_h / dst_h
    labels[:, 3] *= nw / w / maxsize
    labels[:, 4] *= nh / h / maxsize
    if lrflip:
        labels[:, 1] = 1 - labels[:, 1]
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # 原始高，原始宽，缩放后高，缩放后宽，ROI区域左上角x0，ROI区域左上角y0
    h, w, nh, nw, dx, dy = info_img
    # 原始边界框的左上角和右下角坐标
    y1, x1, y2, x2 = box
    # 等比例缩放边界框的宽／高
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    # 基于ROI基底调整边界框左上角坐标，同时进行缩放
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [y1, x1, y1 + box_h, x1 + box_w]
    return label


def preprocess(img, imgsize, jitter, random_placing=False):
    """
    Image preprocess for yolo input
    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position

    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    # 获取图像高/宽
    h, w, _ = img.shape
    # 通道转换: BGR -> RGB
    img = img[:, :, ::-1]
    assert img is not None

    if jitter > 0:
        # 图像抖动
        # add jitter
        # 结果宽 = 抖动因子 * 原图像宽
        dw = jitter * w
        # 结果高 = 抖动因子 * 原图像高
        dh = jitter * h
        # 宽和高的比率
        new_ar = (w + np.random.uniform(low=-dw, high=dw)) \
                 / (h + np.random.uniform(low=-dh, high=dh))
    else:
        # 宽和高的比率
        new_ar = w / h

    if new_ar < 1:
        # 高比宽大
        #
        # 设置高为目标大小
        nh = imgsize
        # 等比例缩放宽
        nw = nh * new_ar
    else:
        # 宽大于等于高
        #
        # 设置宽为目标大小
        nw = imgsize
        # 等比例缩放高
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    if random_placing:
        # 上／下或者左／右随机位置进行填充
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        # 上／下或者左／右等比例填充
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    # 首先将图像缩放到指定大小
    img = cv2.resize(img, (nw, nh))
    # 设置填充图像，目标大小
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    # 设置ROI区域
    sized[dy:dy + nh, dx:dx + nw, :] = img

    # (原始高，原始宽，缩放后高，缩放后宽，ROI区域左上角x0，ROI区域左上角y0)
    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def random_distort(img, hue, saturation, exposure):
    """
    perform random distortion in the HSV color space.
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        hue (float): random distortion parameter.
        saturation (float): random distortion parameter.
        exposure (float): random distortion parameter.
    Returns:
        img (numpy.ndarray)
    """
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue

    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0

    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)

    return img


def get_coco_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )
    coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors
