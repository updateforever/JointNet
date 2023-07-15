import os

import numpy as np
from PIL import Image


def screen(root, subset='train', img_sz=513):
    root = os.path.expanduser(root)  # /root/autodl-tmp/house-2k-seg/VOCdevkit
    base_dir = 'VOCdevkit'
    voc_root = os.path.join(root, base_dir)
    image_dir = os.path.join(voc_root, 'JPEGImages')

    if not os.path.isdir(voc_root):
        raise RuntimeError('Dataset not found.')

    splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
    split_f = os.path.join(splits_dir, subset.rstrip('\n') + '.txt')

    if not os.path.exists(split_f):
        raise ValueError(
            'Wrong image_set entered! Please use image_set="train" '
            'or image_set="trainval" or image_set="val"')

    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

    for i, path in enumerate(images):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if w < img_sz or h < img_sz:
            print('the size of %s is small' % path)
            file_names[i] = '# ' + file_names[i]

    temp_path = os.path.join(splits_dir, 'screen_temp_' + subset.rstrip('\n') + '.txt')
    np.savetxt(temp_path, file_names, delimiter='\n', fmt='%s')
    temp = np.loadtxt(temp_path, dtype=str, comments='#', delimiter='\n')
    new_path = os.path.join(splits_dir, 'screen_' + subset.rstrip('\n') + '.txt')
    np.savetxt(new_path, temp, delimiter='\n', fmt='%s')


if __name__ == '__main__':
    screen('D:/datasets/houseS-2k', subset='train')
    # screen('D:/datasets/houseS-2k', subset='val')
    screen('D:/datasets/houseS-2k', subset='trainval')
