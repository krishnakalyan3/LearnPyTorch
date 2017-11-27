#!/usr/bin/env python3

# Inspiration
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from torch.autograd import Variable
from models import senet


def data_loader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train(epochs):
    net.train()

    for epoch in range(1, epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            data, target = Variable(data), Variable(target)
            outputs = net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, target) in enumerate(test_loader):
        if use_cuda:
            inputs, target = inputs.cuda(), target.cuda()
        inputs, target = Variable(inputs, volatile=True), Variable(target)
        outputs = net(inputs)
        loss = criterion(outputs, target)
        print('{} loss', loss.data[0])

        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()
    print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FeedForward Example')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    train_loader, test_loader = data_loader()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = senet.SENet18()

    if use_cuda:
        total_gpus = torch.cuda.device_count()
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(total_gpus))
        print('{} GPUs available'.format(total_gpus))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train(args.epochs)
