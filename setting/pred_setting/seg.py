from unittest import loader

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

from datasets.house_seg_2k import voc_cmap
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

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
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes + 2, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(opts.device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(opts.device)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if 1:
        transform = T.Compose([
            T.Resize((opts.crop_size, opts.crop_size)),  # T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # else:
    #     transform = T.Compose([
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            img = Image.open(img_path).convert('RGB')
            w, h = np.array(np.shape(img)[0:2])
            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(opts.device)

            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            colorized_preds = opts.decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)

            # F.resize(colorized_preds, [h, w], InterpolationMode.BILINEAR)
            T1 = T.Resize((w, h))
            colorized_preds = T1(colorized_preds)
            # if 0:
            #     pred1 = voc_cmap(pred).astype(np.uint8)
            #     fig = plt.figure()
            #     plt.imshow(img)
            #     plt.axis('off')
            #     plt.imshow(pred1, alpha=0.7)
            #     ax = plt.gca()
            #     ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            #     ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            #     plt.savefig('results/%d_overlay.png' % img_name, bbox_inches='tight', pad_inches=0)
            #     plt.close()

            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))
