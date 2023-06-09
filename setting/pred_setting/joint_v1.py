import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import network
import utils
from torchvision import transforms as T
import torch.nn as nn

from PIL import Image
from glob import glob

import colorsys
import os
import time

import numpy as np
import torch
from PIL import ImageDraw, ImageFont

from network.models.detect.centernet import CenterNet as CenterNet_Resnet50
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_bbox, postprocess


# ---------------------------------------------------#
#   检测图片
# ---------------------------------------------------#
def detect_image(image, img_data, model, opts, crop=False, count=False):
    # ---------------------------------------------------#
    #   计算输入图片的高和宽
    # ---------------------------------------------------#
    image_shape = np.array(np.shape(image)[0:2])

    # 图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
    # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    with torch.no_grad():
        # 将图像输入网络当中进行预测！
        outputs = model(img_data)

        # 利用预测结果进行解码
        outputs = decode_bbox(outputs[0], outputs[1], outputs[2], opts.confidence, opts.device)

        # -------------------------------------------------------#
        #   对于centernet网络来讲，确立中心非常重要。
        #   对于大目标而言，会存在许多的局部信息。
        #   此时对于同一个大目标，中心点比较难以确定。
        #   使用最大池化的非极大抑制方法无法去除局部框
        #   所以我还是写了另外一段对框进行非极大抑制的代码
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        # -------------------------------------------------------#
        results = postprocess(outputs, opts.nms, image_shape, opts.crop_size, opts.letterbox_image, opts.nms_iou)

        # --------------------------------------#
        #   如果没有检测到物体，则返回原图
        # --------------------------------------#
        if results[0] is None:
            return image

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

    # ---------------------------------------------------------#
    #   设置字体与边框厚度
    # ---------------------------------------------------------#
    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * np.shape(image)[1] + 0.5 - 10).astype('int32'))
    thickness = max((np.shape(image)[0] + np.shape(image)[1]) // opts.crop_size, 1)
    # ---------------------------------------------------------#
    #   计数
    # ---------------------------------------------------------#
    if count:
        print("top_label:", top_label)
        classes_nums = np.zeros([opts.num_classes])
        for i in range(opts.num_classes):
            num = np.sum(top_label == i)
            if num > 0:
                print(opts.class_names[i], " : ", num)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)
    # ---------------------------------------------------------#
    #   是否进行目标的裁剪
    # ---------------------------------------------------------#
    if crop:
        for i, c in list(enumerate(top_label)):
            top, left, bottom, right = top_boxes[i]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            dir_save_path = "img_crop"
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            crop_image = image.crop([left, top, right, bottom])
            crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
            print("save crop_" + str(i) + ".png to " + dir_save_path)
    # ---------------------------------------------------------#
    #   图像绘制
    # ---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        predicted_class = opts.class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]

        top, left, bottom, right = box

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=opts.colors[c])
        # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=opts.colors[c])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image


def main(opts):
    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model (all models are 'constructed at network.modeling)

    # det_model
    det_model = network.modeling.__dict__[opts.det_model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    utils.set_bn_momentum(det_model.backbone, momentum=0.01)

    if opts.det_ckpt is not None and os.path.isfile(opts.det_ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.det_ckpt, map_location=torch.device('cpu'))
        det_model.load_state_dict(checkpoint["model_state"])
        det_model = nn.DataParallel(det_model)
        det_model.to(opts.device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("No ck for predict")

        det_model = nn.DataParallel(det_model)
        det_model.to(opts.device)

    # seg_model
    seg_model = network.modeling.__dict__[opts.seg_model](num_classes=opts.num_classes + 2, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.seg_model:
        network.convert_to_separable_conv(seg_model.classifier)
    utils.set_bn_momentum(seg_model.backbone, momentum=0.01)

    if opts.seg_ckpt is not None and os.path.isfile(opts.seg_ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.seg_ckpt, map_location=torch.device('cpu'))
        seg_model.load_state_dict(checkpoint["model_state"])
        seg_model = nn.DataParallel(seg_model)
        seg_model.to(opts.device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        seg_model = nn.DataParallel(seg_model)
        seg_model.to(opts.device)

    # 计算总的类的数量  classes_path = 'model_data/voc_house6.txt'
    opts.class_names, opts.num_classes = get_classes(opts.classes_path)

    # 画框设置不同的颜色
    hsv_tuples = [(x / opts.num_classes, 1., 1.) for x in range(opts.num_classes)]
    opts.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    opts.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), opts.colors))

    # trans
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    # pre
    with torch.no_grad():
        seg_model = seg_model.eval()
        det_model = det_model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            img = Image.open(img_path).convert('RGB')  # cvtColor
            re_image = img.resize((opts.crop_size, opts.crop_size), Image.BICUBIC)  # resize
            img_data = transform(re_image).unsqueeze(0)  # To tensor of NCHW
            img_data = img_data.to(opts.device)
            # seg
            pred = seg_model(img_data).max(1)[1].cpu().numpy()[0]  # HW
            colorized_preds = opts.decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)  # seg_pred_img = Image.fromarray(colorized_preds)

            plt.imshow(re_image)
            plt.axis('off')
            plt.imshow(colorized_preds, alpha=0.7)
            ax = plt.gca()
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            plt.savefig(os.path.join(opts.save_val_results_to, '%s_overlay.png' % img_name), bbox_inches='tight', pad_inches=0)
            plt.close()

            mask_img = Image.open(os.path.join(opts.save_val_results_to, '%s_overlay.png' % img_name)).convert('RGB')  # cvtColor
            mask_img = mask_img.resize((opts.crop_size, opts.crop_size), Image.BICUBIC)  # resize

            # det
            mask_img_data = transform(mask_img).unsqueeze(0)
            mask_img_data = mask_img_data.to(opts.device)
            if opts.crop_img:
                pred_img = detect_image(re_image, mask_img_data, det_model, opts=opts, count=True)  # resize img
            else:
                # pred_img = detect_image(img, mask_img_data, det_model, opts=opts, count=True)  # origin img
                pred_img = detect_image(mask_img, mask_img_data, det_model, opts=opts, count=True)
            # save
            if opts.save_val_results_to:
                pred_img.save(os.path.join(opts.save_val_results_to, img_name + '.jpg'))

            print('done')
