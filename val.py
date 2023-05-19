import importlib
import torch.distributed as dist

import network
import os
import random
import argparse
import numpy as np

import torch


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='D:/datasets/house-2k',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='house2k',
                        choices=['voc', 'cityscapes', 'house2k'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=6,
                        help="num classes (default: None)")
    parser.add_argument("--classes_path", type=str,
                        default='D:/DPcode/centernet-pytorch-main/model_data/voc_house6.txt')
    parser.add_argument("--crop_size", type=int, default=513)  # 385  513

    # Model Options
    parser.add_argument("--mode", type=str, default='det', choices=['det', 'seg', 'joint'], help='model class')
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and
                              not (name.startswith("__") or name.startswith('_'))
                              and callable(network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='centernet', choices=available_models, help='model name')
    parser.add_argument("--ckpt", default=None, type=str, help="restore from checkpoint")

    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Val Options
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--val_path", type=str, default='D:/DPcode/JointNet/result', help="")
    parser.add_argument("--crop_val", action='store_true', default=False, help='crop validation (default: False)')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    # parser.add_argument("--", type=float, default=0.5, help="")
    # parser.add_argument("--", type=float, default=0.5, help="")

    # DETECT
    parser.add_argument("--MINOVERLAP", type=float, default=0.5, help="MINOVERLAP")
    parser.add_argument("--confidence", type=float, default=0.02, help="confidence")
    parser.add_argument("--nms_iou", type=float, default=0.5, help="nms_iou")
    parser.add_argument("--nms", action='store_true', default=True, help="whether using nms")
    parser.add_argument("--score_threhold", type=float, default=0.5, help="score_threhold")

    return parser


def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts.local_rank = 0
    print("Device: %s" % opts.device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    run = importlib.import_module('setting.val_setting.{}'.format(opts.mode))
    run.main(opts)


if __name__ == '__main__':
    main()
