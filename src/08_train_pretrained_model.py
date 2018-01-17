#!/usr/bin/env python3

import torch
from torchvision import transforms
from models.basic_models import Net
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from other.utils import load_model


MNIST_DATA = '../data/MNIST'


def data_loader():
    train_dataset = dsets.MNIST(root=MNIST_DATA,
                                train=True,
                                transform=transforms.Compose([transforms.ToTensor()]),

                                download=True)

    test_dataset = dsets.MNIST(root=MNIST_DATA,
                               train=False,
                               transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False)
    return train_loader, test_loader


def train(epochs):
    model.train()

    for epoch in range(1, epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = Variable(data), Variable(target)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    model = Net()
    model = load_model(model, '../data/models/mnist_model.pkl')
    train_loader, test_loader = data_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    train(1)
