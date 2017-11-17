#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.basic_models import Net
from other.logger import Logger, to_np
from utils import tensorboard_logger
import argparse

MNIST_DATA = '../data/MNIST'


def data_loader(batch_size):
    train_dataset = dsets.MNIST(root=MNIST_DATA,
                                train=True,
                                transform=transforms.Compose([transforms.ToTensor()]),
                                download=True)
    test_dataset = dsets.MNIST(root=MNIST_DATA,
                               train=False,
                               transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def train(epochs):
    model.train()
    for epoch in range(1, epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(outputs, 1)
            accuracy = (target == argmax.squeeze()).float().mean()

            if batch_idx % 50 == 0:
                tensorboard_logger(loss, accuracy, epoch, model, data)
                print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()
    print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FeedForward Example')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    args = parser.parse_args()

    train_loader, test_loader = data_loader(args.batch_size)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train(args.epochs)

