import importlib
import torch.distributed as dist

import network
import os
import random
import argparse
import numpy as np

import torch
import warnings


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='D:/datasets/house2k-master',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='house2k',
                        choices=['voc', 'cityscapes', 'house2k'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default='',
                        help="num classes (default: None)")  # 8 3

    # Model Options
    parser.add_argument("--model_opt", type=str, default='det',
                        choices=['det', 'seg', 'joint'], help='model class')
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='centernet',
                        choices=available_models, help='model name')

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--train_epochs", type=int, default=150,
                        help="epoch number (default: 200)")
    parser.add_argument("--optimizer", type=str, default='', help="optimizer")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=12,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=10,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)  # 512  640

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='focal_loss',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    # parser.add_argument("--print_interval", type=int, default=10,
    #                     help="print interval of loss (default: 10)")
    # parser.add_argument("--val_interval", type=int, default=100,
    #                     help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--distributed", action='store_true', default=False,
                        help="distributed")

    # #segment options
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # #detect options
    parser.add_argument("--model_path", type=str, default='', help="model_path for train")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',  # --enable_vis --vis_port 28333
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    if opts.distributed:
        ngpus_per_node = torch.cuda.device_count()
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    opts.device = device
    opts.local_rank = local_rank
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    run = importlib.import_module('setting.train_setting.{}'.format(opts.model_opt))
    run.main(opts)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
