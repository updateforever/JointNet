import datetime
import os
import xml.etree.ElementTree as ET

from utils.utils_map import get_coco_map, get_map
from tqdm import tqdm
import network
import utils
from torchvision import transforms as T
import torch.nn as nn
from PIL import Image
import os

import numpy as np
import torch
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_bbox, postprocess


def get_map_txt(img_org, img_data, image_id, model, map_out_path, opts):
    if os.path.exists(os.path.join(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"))):
        return
    f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
    image_shape = np.array(np.shape(img_org)[0:2])
    with torch.no_grad():
        # ---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        # ---------------------------------------------------------#
        outputs = model(img_data)
        # -----------------------------------------------------------#
        #   利用预测结果进行解码
        # -----------------------------------------------------------#
        outputs = decode_bbox(outputs[0], outputs[1], outputs[2], opts.confidence, opts.device)
        results = postprocess(outputs, opts.nms, image_shape, opts.crop_size - 1, letterbox_image=False,
                              nms_thres=opts.nms_iou)

        if results[0] is None:
            return

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

    for i, c in list(enumerate(top_label)):
        predicted_class = opts.class_names[int(c)]
        box = top_boxes[i]
        score = str(top_conf[i])

        top, left, bottom, right = box

        if predicted_class not in opts.class_names:
            continue

        f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

    f.close()
    return


def main(opts):
    """
        Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
        默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

        受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
        因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    """
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M')
    map_out_path = os.path.join(opts.val_path, opts.mode, 'val', str(time_str))  # 结果存放路径
    data_root = os.path.join(opts.data_root, 'VOCdevkit')
    image_ids = open(os.path.join(data_root, "VOC2012/ImageSets/Main/test_coco.txt")).read().strip().split()
    if image_ids[0].find('jpg'):
        for i, image_id in enumerate(image_ids):
            image_ids[i] = image_id.replace('.jpg', '')

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))

    opts.class_names, opts.num_classes = get_classes(opts.classes_path)

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(opts.device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("No ck for predict")

        model = nn.DataParallel(model)
        model.to(opts.device)

    if opts.crop_val:
        transform = T.Compose([
            # T.Resize((opts.crop_size - 1, opts.crop_size - 1)),  # T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    with torch.no_grad():
        model = model.eval()
        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_root, "VOC2012/JPEGImages/" + image_id + ".jpg")
            img = Image.open(image_path).convert('RGB')
            # resize
            re_image = img.resize((opts.crop_size - 1, opts.crop_size - 1), Image.BICUBIC)
            # transform
            img_data = transform(re_image).unsqueeze(0)  # To tensor of N, C, H, W
            img_data = img_data.to(opts.device)

            get_map_txt(img, img_data, image_id, model, map_out_path, opts=opts)  # HW

        print("Get predict result done.")

        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(data_root, "VOC2012/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in opts.class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

        print("Get map.")
        get_map(opts.MINOVERLAP, True, score_threhold=opts.score_threhold, path=map_out_path)
        print("Get map done.")

        print("Get COCO map.")
        get_coco_map(class_names=opts.class_names, path=map_out_path)
        print("Get COCO map done.")
