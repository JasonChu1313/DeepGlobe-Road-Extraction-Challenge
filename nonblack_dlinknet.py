import pandas as pd
import os
import csv
import os
from glob import glob
from skimage.io import imread, imsave
import numpy as np

from glob import glob
import os

path_to_train = './data/train/'
glob_train_imgs = os.path.join(path_to_train, '*_sat.jpg')
glob_train_masks = os.path.join(path_to_train, '*_msk.png')

train_img_paths = glob(glob_train_imgs)
train_mask_paths = glob(glob_train_masks)
print(train_img_paths[:10])
print(train_mask_paths[:10])

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


def get_img_id(img_path):
    img_basename = os.path.basename(img_path)
    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
    return img_id


def img_gen(img_paths, img_size=(512, 512)):
    # Iterate over all image paths
    for img_path in img_paths:
        img_id = get_img_id(img_path)
        mask_path = os.path.join(path_to_train, img_id + '_msk.png')

        img = imread(img_path) / 255
        mask = rgb2gray(imread(mask_path))

        #     img = resize(img, img_size, preserve_range = True)
        #     mask = resize(mask, img_size, mode='constant', preserve_range = True)

        mask = (mask >= 0.5).astype(float)
        mask = np.reshape(mask, (512, 512, 1))
        yield img, mask


def get_non_outlier_data(train_img_paths):
    train_path_without_outlier = []
    for index, image_path in enumerate(train_img_paths):
        if index % 500 == 0:
            print(index)
        img_id = get_img_id(image_path)
        mask_path = os.path.join('./data/train/', img_id + '_msk.png')
        mask = rgb2gray(imread(mask_path))
        if len(np.where(mask.flatten() != 0)[0]) < 800:
            os.remove(path_to_train + img_id + '_sat.jpg')
            os.remove(path_to_train + img_id + '_msk.png')
    return train_path_without_outlier


"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def default_loader(id, root):
    img = cv2.imread(os.path.join(root, '{}_sat.jpg').format(id))
    mask = cv2.imread(os.path.join(root + '{}_msk.png').format(id), cv2.IMREAD_GRAYSCALE)

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)

    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


class ImageFolder(data.Dataset):
    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img, mask

    def __len__(self):
        return len(self.ids)


import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print >> mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr)
        print
        'update learning rate: %f -> %f' % (self.old_lr, new_lr)
        self.old_lr = new_lr

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tqdm import tqdm
import cv2
import os
import numpy as np

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from loss import dice_bce_loss

SHAPE = (512, 512)
ROOT = '../Satellite-Segmentation/data/train/'
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-8], imagelist)
NAME = 'log01_dink34'
BATCHSIZE_PER_CARD = 4

solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

mylog = open('logs/' + NAME + '.log', 'w')
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in tqdm(data_loader_iter):
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print >> mylog, '********'
    print >> mylog, 'epoch:', epoch, '    time:', int(time() - tic)
    print >> mylog, 'train_loss:', train_epoch_loss
    print >> mylog, 'SHAPE:', SHAPE
    print
    '********'
    print
    'epoch:', epoch, '    time:', int(time() - tic)
    print
    'train_loss:', train_epoch_loss
    print
    'SHAPE:', SHAPE

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '.th')
    if no_optim > 6:
        print >> mylog, 'early stop at %d epoch' % epoch
        print
        'early stop at %d epoch' % epoch
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

print >> mylog, 'Finish!'
print
'Finish!'
mylog.close()