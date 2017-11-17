#!/usr/bin/env python3
import torch
import argparse
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from models.wideresnet import WideResNet

CIFAR_DATA = '../data/CIFAR'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FeedForward Example')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_dataset = dsets.CIFAR10(root=CIFAR_DATA,
                                  train=True,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor(), normalize]),

                                  download=True)

    test_dataset = dsets.CIFAR10(root=CIFAR_DATA,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(), normalize]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False)

    model = WideResNet(int(28), 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                epoch + 1, args.epochs, i + 1, len(train_dataset) // args.batch_size,
                loss.data[0]))

    # Test
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images, volatile=True)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # print('Accuracy of the network on the %d test images: %d %%' % (len(test_loader.dataset), 100 * correct / total))
        #
        # for i in range(len(labels)):
        #
        #     # Print Incorrect Images
        #     if labels[i] != predicted[i]:
        #         current_image = images.data.numpy()[i].reshape(-1, 28)
        #         plt.clf()
        #         plt.imshow(current_image, cmap='gray_r', )
        #         plt.show(block=False)
        #         plt.pause(0.10)
        #         print('True Label {}, Predict Label {}'.format(labels[i], predicted[i]))

    print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))
