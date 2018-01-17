#!/usr/bin/env python3
from other.logger import Logger
import os
import torch.utils.data as data
import torch
from PIL import Image
import sys
import psutil


def tensorboard_logger(loss, accuracy, epoch, net=None, images=None):
    logger = Logger('../logs/06_tensorboard_1')

    # (1) Log the scalar values
    info = {
        'loss': loss.data[0],
        'accuracy': accuracy.data[0]
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), epoch)
        logger.histo_summary(tag + '/grad', to_np(value.grad), epoch)

    # (3) Log the images
    if images is not None:
        info = {
            'images': to_np(images.view(-1, 28, 28)[:10])
        }

        for tag, images in info.items():
            logger.image_summary(tag, images, epoch + 1)


def to_np(x):
    return x.data.cpu().numpy()

class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)


def cpu_stats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    return model


def resnext50(pre):
    return load_pre(pre, resnext_50_32x4d, 'resnext_50_32x4d')


def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m