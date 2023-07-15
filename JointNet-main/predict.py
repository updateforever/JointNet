import importlib

from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes, house2k_seg
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'house2k'], help='Name of training set')
    parser.add_argument("--classes_path", type=str, default='D:/datasets/house2k-master/class.txt')
    # D:/datasets/house-2k/VOCdevkit/VOC2012/JPEGImages/3.jpg

    # Model Options
    parser.add_argument("--mode", type=str, default='joint_v1', help='model class')
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')

    # Deeplab Options
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--save_overlay_img", type=bool, default=True)

    # Detect Options
    parser.add_argument("--letterbox_image", type=bool, default=False,
                        help='用于控制是否使用letterbox_image对输入图像进行不失真的resize')
    parser.add_argument("--nms", action='store_true', default=True, help="是否进行非极大抑制，可以根据检测效果自行选择")
    parser.add_argument("--nms_iou", type=float, default=0.3)
    parser.add_argument("--confidence", type=float, default=0.3)  #\

    # JointNet Options
    parser.add_argument("--det_model", type=str, default='centernet_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--seg_model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--det_ckpt", default='', type=str, help="resume from checkpoint")
    # D:/DPcode/JointNet/network/524-best_centernet_resnet50_house-2k.pth
    parser.add_argument("--seg_ckpt", default='D:/DPcode/JointNet/checkpoints/2023-04-17_17-57/'
                                              'best_deeplabv3plus_resnet50_house-2k_os16-0418.pth',
                        type=str, help="resume from checkpoint")

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")
    parser.add_argument("--crop_img", action='store_true', default=False,
                        help='whether outputting cropped img (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        opts.decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        opts.decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'houseS2k':
        opts.num_classes = 8
        opts.decode_fn = house2k_seg.decode_target
    elif opts.dataset.lower() == 'house2k':
        opts.num_classes = 5
        opts.decode_fn = house2k_seg.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % opts.device)

    run = importlib.import_module('setting.pred_setting.{}'.format(opts.mode))
    run.main(opts)


if __name__ == '__main__':
    main()
