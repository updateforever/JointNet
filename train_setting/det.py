import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, house2k_seg
from datasets.house_det_2k import CenternetDataset, centernet_dataset_collate
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import time


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    # if opts.dataset == 'voc':
    #     train_transform = et.ExtCompose([
    #         # et.ExtResize(size=opts.crop_size),
    #         et.ExtRandomScale((0.5, 2.0)),
    #         et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
    #         et.ExtRandomHorizontalFlip(),
    #         et.ExtToTensor(),
    #         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    #     if opts.crop_val:
    #         val_transform = et.ExtCompose([
    #             et.ExtResize(opts.crop_size),
    #             et.ExtCenterCrop(opts.crop_size),
    #             et.ExtToTensor(),
    #             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225]),
    #         ])
    #     else:
    #         val_transform = et.ExtCompose([
    #             et.ExtToTensor(),
    #             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225]),
    #         ])
    #     train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
    #                                 image_set='train', download=opts.download, transform=train_transform)
    #     val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
    #                               image_set='val', download=False, transform=val_transform)
    #
    # if opts.dataset == 'house-2k':
    #     train_transform = et.ExtCompose([
    #         et.ExtResize((513, 513)),
    #         et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    #         et.ExtRandomHorizontalFlip(),
    #         et.ExtToTensor(),
    #         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     val_transform = et.ExtCompose([
    #         et.ExtResize((513, 513)),
    #         et.ExtToTensor(),
    #         et.ExtNormalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])
    #
    #     train_dst = house2k_seg(root=opts.data_root, image_set='train', transform=train_transform,
    #                             img_sz=opts.crop_size)
    #     val_dst = house2k_seg(root=opts.data_root, image_set='val', transform=val_transform, img_sz=opts.crop_size)

    train_annotation_path = r'D:\datasets\house-2k\VOCdevkit\VOC2012\ImageSets\2012_train.txt'
    val_annotation_path = r'D:\datasets\house-2k\VOCdevkit\VOC2012\ImageSets\2012_val.txt'
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    train_dst = CenternetDataset(train_lines, [opts.crop_size - 1, opts.crop_size - 1], opts.num_classes, train=True)
    val_dst = CenternetDataset(val_lines, [opts.crop_size - 1, opts.crop_size - 1], opts.num_classes, train=False)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main(opts):
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'house-2k':
        opts.num_classes = 6  # 6

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_epochs": cur_epochs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=False, num_workers=2,
        drop_last=True, collate_fn=centernet_dataset_collate)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2, collate_fn=centernet_dataset_collate,
        drop_last=True)

    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    if opts.model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if opts.local_rank == 0:
            print('Load weights {}.'.format(opts.model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opts.model_path, map_location=opts.device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if opts.local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # Set up optimizer
    for n, p in model.named_parameters():
        if 'backbone' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    optimizer = torch.optim.Adam(params=[{'params': model.backbone.parameters(), 'lr': 0.5 * opts.lr},
                                         {'params': model.decoder.parameters(), 'lr': opts.lr},
                                         {'params': model.head.parameters(), 'lr': opts.lr}
                                         ], lr=opts.lr, betas=(0.9, 0.999), weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion = utils.get_loss(opts.loss_type)
    criterion = utils.DETloss()
    ck_path = os.path.join(os.path.abspath('.'), 'checkpoints', time.strftime("%Y-%m-%d_%H-%M", time.localtime()))
    utils.mkdir(ck_path)
    # Restore
    cur_epochs = 0
    interval_loss = 0
    best_score = 0.0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(opts.device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epochs = checkpoint["cur_epochs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(opts.device)

    # tensorboard
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%H_%M')
    log_dir = os.path.join('runs/train', str(time_str))
    writer = SummaryWriter(log_dir=log_dir)
    '''
        tensorboard 
    '''

    # ==========   Train Loop   ==========#
    best_loss = 1e8
    # =====  Train  =====
    for i in range(0, opts.train_epochs):
        cur_epochs += 1
        cur_itrs = 0
        cur_itrs_val = 0

        total_r_loss = 0
        total_c_loss = 0
        total_loss = 0
        val_loss = 0

        if cur_epochs >= 50:
            # unfreezn
            for n, p in model.parameters():
                if 'backbone.layer4' in n:
                    p.requires_grad = True

        model.train()
        for j, batch in enumerate(train_loader):

            total_itrs = len(train_loader)
            cur_itrs = j + 1

            batch = [ann.to(opts.device) for ann in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            hm, wh, offset = model(batch_images)  # (b,cl,h,w)

            loss, c_loss, r_loss = criterion(hm, wh, offset, batch)
            total_loss += loss.item()
            total_c_loss += c_loss.item()
            total_r_loss += r_loss.item()

            loss.requires_grad_()
            # 三件套
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cur_itrs % 5 == 0 or cur_itrs == total_itrs:  # 隔5批次打印一次
                print("Epoch %d, Itrs %d/%d, Loss=%f, ClLoss=%f, ReLoss=%f" % (
                    cur_epochs, cur_itrs, total_itrs, loss, c_loss, r_loss))

        # lr scheduler
        scheduler.step()
        cr_lr = optimizer.param_groups[0]['lr']

        # log train
        writer.add_scalar(tag="train/total_loss", scalar_value=total_loss / len(train_loader), global_step=cur_epochs)
        writer.add_scalar(tag="train/cl_loss", scalar_value=total_c_loss / len(train_loader), global_step=cur_epochs)
        writer.add_scalar(tag="train/re_loss", scalar_value=total_r_loss / len(train_loader), global_step=cur_epochs)
        writer.add_scalar(tag="train/lr", scalar_value=cr_lr, global_step=cur_epochs)
        '''
        if vis is not None:
            vis.vis_scalar('Loss', cur_epochs, total_loss / cur_itrs)
            vis.vis_scalar('CL_Loss', cur_epochs, total_c_loss / cur_itrs)
            vis.vis_scalar('RE_Loss', cur_epochs, total_r_loss / cur_itrs)
        '''

        # val and save pth
        print("validation...")
        model.eval()
        for j, batch in enumerate(val_loader):
            cur_itrs_val += 1
            with torch.no_grad():
                batch = [ann.to(opts.device) for ann in batch]
                batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

                hm, wh, offset = model(batch_images)
                loss, c_loss, r_loss = criterion(hm, wh, offset, batch)

                val_loss += loss

        # log val
        writer.add_scalar(tag="loss/val", scalar_value=val_loss / len(val_loader), global_step=cur_epochs)
        print('Finish Validation')
        print('Epoch:' + str(cur_epochs) + '/' + str(opts.train_epochs))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / len(train_loader), val_loss / cur_itrs_val))
        # save best model
        if val_loss / cur_itrs_val < best_loss:
            best_loss = val_loss
            save_ckpt(os.path.join(ck_path, 'best_%s_%s.pth' % (opts.model, opts.dataset)))
        # save 5x.pth
        if cur_epochs % 5 == 0:
            save_ckpt(os.path.join(ck_path, 'weights_%s_%s_ep%d.pth' % (
                opts.model, opts.dataset, cur_epochs)))

    writer.close()
    print('train done')
