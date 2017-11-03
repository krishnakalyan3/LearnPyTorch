#!/usr/bin/env python3
from logger import Logger

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
