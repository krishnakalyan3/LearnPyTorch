#!/usr/bin/env python3

import torch
import torchvision
from torchvision import transforms

CIFAR_DATA = '../data/CIFAR'

if __name__ == '__main__':
    normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=CIFAR_DATA, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])), batch_size=128, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=CIFAR_DATA, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])), batch_size=128, shuffle=False, num_workers=2)