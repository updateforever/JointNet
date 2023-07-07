import datetime

import wandb
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
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import time


# def get_argparser():
#     parser = argparse.ArgumentParser()
#
#     # Datset Options
#     parser.add_argument("--data_root", type=str, default='D:/datasets/houseS-2k',
#                         help="path to Dataset")
#     parser.add_argument("--dataset", type=str, default='house-2k',
#                         choices=['voc', 'cityscapes', 'house-2k'], help='Name of dataset')
#     parser.add_argument("--num_classes", type=int, default=None,
#                         help="num classes (default: None)")  # 8
#
#     # Deeplab Options
#     available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
#                               not (name.startswith("__") or name.startswith('_')) and callable(
#         network.modeling.__dict__[name])
#                               )
#     parser.add_argument("--model", type=str, default='centernet',
#                         choices=available_models, help='model name')
#     parser.add_argument("--separable_conv", action='store_true', default=False,
#                         help="apply separable conv to decoder and aspp")
#     parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
#
#     # Train Options
#     parser.add_argument("--test_only", action='store_true', default=False)
#     parser.add_argument("--save_val_results", action='store_true', default=False,
#                         help="save segmentation results to \"./results\"")
#     parser.add_argument("--train_epochs", type=int, default=300,
#                         help="epoch number (default: 200)")
#     parser.add_argument("--total_itrs", type=int, default=30000,
#                         help="epoch number (default: 30k)")
#     parser.add_argument("--lr", type=float, default=0.01,
#                         help="learning rate (default: 0.01)")
#     parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
#                         help="learning rate scheduler policy")
#     parser.add_argument("--step_size", type=int, default=10000)
#     parser.add_argument("--crop_val", action='store_true', default=False,
#                         help='crop validation (default: False)')
#     parser.add_argument("--batch_size", type=int, default=16,
#                         help='batch size (default: 16)')
#     parser.add_argument("--val_batch_size", type=int, default=4,
#                         help='batch size for validation (default: 4)')
#     parser.add_argument("--crop_size", type=int, default=513)  # 385  513
#
#     parser.add_argument("--ckpt", default=None, type=str,
#                         help="restore from checkpoint")
#     parser.add_argument("--continue_training", action='store_true', default=False)
#
#     parser.add_argument("--loss_type", type=str, default='focal_loss',
#                         choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
#     parser.add_argument("--gpu_id", type=str, default='0',
#                         help="GPU ID")
#     parser.add_argument("--weight_decay", type=float, default=1e-4,
#                         help='weight decay (default: 1e-4)')
#     parser.add_argument("--random_seed", type=int, default=1,
#                         help="random seed (default: 1)")
#     # parser.add_argument("--print_interval", type=int, default=10,
#     #                     help="print interval of loss (default: 10)")
#     # parser.add_argument("--val_interval", type=int, default=100,
#     #                     help="epoch interval for eval (default: 100)")
#     parser.add_argument("--download", action='store_true', default=False,
#                         help="download datasets")
#
#     # PASCAL VOC Options
#     parser.add_argument("--year", type=str, default='2012',
#                         choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')
#
#     # Visdom options
#     parser.add_argument("--enable_vis", action='store_true', default=False,
#                         help="use visdom for visualization")
#     parser.add_argument("--vis_port", type=str, default='13570',  # --enable_vis --vis_port 28333
#                         help='port for visdom')
#     parser.add_argument("--vis_env", type=str, default='main',
#                         help='env for visdom')
#     parser.add_argument("--vis_num_samples", type=int, default=8,
#                         help='number of samples for visualization (default: 8)')
#     return parser
#

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    if opts.dataset == 'house2k':
        train_transform = et.ExtCompose([
            et.ExtResize((opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize((opts.crop_size, opts.crop_size)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = house2k_seg(root=opts.data_root, image_set='trainval_coco', transform=train_transform)
        val_dst = house2k_seg(root=opts.data_root, image_set='test_coco', transform=val_transform)

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
    # Setup visualization
    name = opts.model + '--' + str(datetime.datetime.strftime(datetime.datetime.now(), '%m-%d_%H-%M'))
    wb = wandb.init(project='seg',
                    name=name,
                    config=opts,
                    id='jay')

    # # Setup random seed
    # torch.manual_seed(opts.random_seed)
    # np.random.seed(opts.random_seed)
    # random.seed(opts.random_seed)

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
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes + 2, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes + 2)  # 流指标？

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    ck_path = os.path.join(os.path.abspath(''), 'result', opts.model_opt, 'train',
                           time.strftime("%Y-%m-%d_%H-%M", time.localtime()))
    utils.mkdir(ck_path)
    # Restore
    cur_epochs = 0
    interval_loss = 0
    best_score = 0.0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
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

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=opts.device, metrics=metrics,
            ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    # Print arguments
    # for k, v in sorted(vars(opts).items()):
    #     print(k, '=', v)

    # # tensorboard
    # time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%H_%M')
    # log_dir = os.path.join('runs/train', str(time_str))
    # writer = SummaryWriter(log_dir=log_dir)
    # writer.add_text('Info Of Training', 'train for segment | {}'.format(opts.model))

    # =====  Train  =====
    for i in range(0, opts.train_epochs):
        cur_epochs += 1
        cur_itrs = 0

        total_loss = 0
        val_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            total_itrs = len(train_loader)
            cur_itrs = i + 1

            images = images.to(opts.device, dtype=torch.float32)
            labels = labels.to(opts.device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)  # (b,cl,h,w)
            loss = criterion(outputs, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if cur_itrs % 5 == 0 or cur_itrs == total_itrs:  # 隔5批次打印一次log
                interval_loss = interval_loss / 5
                print("Epoch %d, Itrs %d/%d, Loss=%f" % (cur_epochs, cur_itrs, total_itrs, interval_loss))
                interval_loss = 0.0

        scheduler.step()
        cr_lr = optimizer.param_groups[1]['lr']

        # val and save pth
        if cur_epochs % 5 == 0:
            save_ckpt(os.path.join(ck_path, 'weights_%s_%s_os%d_ep%d.pth' % (
                opts.model, opts.dataset, opts.output_stride, cur_epochs)))
        print("validation...")
        model.eval()
        metrics.reset()
        ret_samples = []
        if opts.save_val_results:
            if not os.path.exists('results'):
                os.mkdir('results')
            denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            img_id = 0

        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(val_loader)):

                images = images.to(opts.device, dtype=torch.float32)
                labels = labels.to(opts.device, dtype=torch.long)

                outputs = model(images)
                val_loss += criterion(outputs, labels)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()

                metrics.update(targets, preds)
                if vis_sample_id is not None and i in vis_sample_id:  # get vis samples
                    ret_samples.append(
                        (images[0].detach().cpu().numpy(), targets[0], preds[0]))

                if opts.save_val_results:
                    for i in range(len(images)):
                        image = images[i].detach().cpu().numpy()
                        target = targets[i]
                        pred = preds[i]

                        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = val_loader.dataset.decode_target(target).astype(np.uint8)
                        pred = val_loader.dataset.decode_target(pred).astype(np.uint8)

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

        # log
        # writer.add_scalar(tag="train/total_loss", scalar_value=total_loss / len(train_loader),
        #                   global_step=cur_epochs)
        # writer.add_scalar(tag="lr", scalar_value=cr_lr, global_step=cur_epochs)
        # class_iou = {}
        # val_score = metrics.get_results()
        # # score_str = metrics.to_str(val_score)
        # writer.add_scalar(tag="val/loss", scalar_value=val_loss / len(val_loader), global_step=cur_epochs)
        # writer.add_scalar(tag="val/Overall Acc", scalar_value=val_score['Overall Acc'], global_step=cur_epochs)
        # writer.add_scalar(tag="val/Mean IoU", scalar_value=val_score['Mean IoU'], global_step=cur_epochs)
        # class_iou['wall'] = val_score['Class IoU'][0]
        # class_iou['window'] = val_score['Class IoU'][1]
        # class_iou['1door'] = val_score['Class IoU'][2]
        # class_iou['2door'] = val_score['Class IoU'][3]
        # class_iou['3door'] = val_score['Class IoU'][4]
        # class_iou['pdoor'] = val_score['Class IoU'][5]
        # writer.add_scalars(main_tag="val/Class IoU", tag_scalar_dict=class_iou, global_step=cur_epochs)
        val_score = metrics.get_results()
        wb.log({'loss': total_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'Overall Acc': val_score['Overall Acc'],
                'Mean IoU': val_score['Mean IoU'],
                'learning rate': cr_lr,
                'epoch': cur_epochs,
                })

        # for k, (img, target, lbl) in enumerate(ret_samples):
        #     img = (denorm(img) * 255).astype(np.uint8)
        #     target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
        #     lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
        #     concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
        #     writer.add_image("val/img", img_tensor=concat_img, global_step=cur_epochs)

        if val_score['Mean IoU'] > best_score:  # save best model
            best_score = val_score['Mean IoU']
            save_ckpt(os.path.join(ck_path, 'best_%s_%s_os%d.pth' %
                                   (opts.model, opts.dataset, opts.output_stride)))
    print('train done')

