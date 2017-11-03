#!/usr/bin/env python3
# Please download the housing dataset from https://www.kaggle.com/apratim87/housingdata

import matplotlib
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.basic_models import LogisticRegression
import torchvision.transforms as transforms
import torchvision.datasets as dsets


MNIST_DATA = '../data/MNIST'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FeedForward Example')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()

    train_dataset = dsets.MNIST(root=MNIST_DATA,
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root=MNIST_DATA,
                               train=False,
                               transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False)

    model = LogisticRegression(input_size=784, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'% (epoch + 1, args.epochs, i + 1, len(train_dataset) // args.batch_size,
                                                              loss.data[0]))

    # Test
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        for i in range(len(labels)):

            # Print Incorrect Images
            if labels[i] != predicted[i]:
                current_image = images.data.numpy()[i].reshape(-1, 28)
                plt.clf()
                plt.imshow(current_image, cmap='gray_r', )
                plt.show(block=False)
                plt.pause(0.10)
                print('True Label {}, Predict Label {}'.format(labels[i], predicted[i]))

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
