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
    parser.add_argument("--data_root", type=str, default='D:/datasets/houseS-2k',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='house-2k',
                        choices=['voc', 'cityscapes', 'house-2k'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")  # 8

    # Model Options
    parser.add_argument("--model_opt", type=str, default='det',
                        choices=['det', 'seg', 'joint'], help='model class')
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
        network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='centernet',
                        choices=available_models, help='model name')

    # Val Options
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")


def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    if opts.distributed:
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

    run = importlib.import_module('val_setting.{}'.format(opts.model_opt))
    run.main(opts)


if __name__ == '__main__':
    main()
