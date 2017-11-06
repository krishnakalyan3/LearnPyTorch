#!/usr/bin/env python3
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms
import torchvision.models as models
import argparse
from other.utils import TestImageFolder
from PIL import Image


CATS_DOGS = '../data/dogscats/'


def data_loader(batch_size):

    transform_compose = transforms.Compose([transforms.RandomSizedCrop(224),
                                            #transforms.Scale(256),
                                            #transforms.Lambda(lambda img: img.resize((224, 224), Image.BILINEAR)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

    train_dataset = ImageFolder(root=CATS_DOGS + 'train', transform=transform_compose)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, **kwargs)

    val_dataset = ImageFolder(root=CATS_DOGS + 'valid', transform=transform_compose)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True, **kwargs)

    test_dataset = TestImageFolder(root=CATS_DOGS + 'test1', transform=transform_compose)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False, **kwargs)

    print('classes {}'.format(train_dataset.classes))
    return train_loader, val_loader, test_loader


def train(epochs):
    model.train()

    for epoch in range(1, epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def valid():
    model.eval()
    correct = 0
    total = 0
    for data, target in val_loader:
        data, target = Variable(data.cuda()), Variable(target)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)

        total += target.size(0)
        correct += (predicted == target.data).sum()

    print('Accuracy of the network on the %d valid images: %d %%' % (total, 100 * correct / total))


def test():
    model.eval()
    for data, target in test_loader:
        image_var = Variable(data, volatile=True)
        y_pred = model(image_var)
        print(y_pred.data)
        # All predictions


if __name__ == '__main__':
    backend = ['cpu', 'gpu']

    parser = argparse.ArgumentParser(description='PyTorch Cats vs Dogs Example')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--backend', default='cpu', type=str, help='Type of backend', choices=backend)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader, val_loader, test_loader = data_loader(args.batch_size)

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model2 = nn.Sequential(nn.Linear(512, 128),
                           nn.ReLU(True),
                           nn.Dropout(),
                           nn.Linear(128, 64),
                           nn.ReLU(True),
                           nn.Dropout(),
                           nn.Linear(64, 2))

    # model.fc = nn.Linear(512, 2)
    # or use model2
    model.fc = model2
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    only_grad_param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(only_grad_param, lr=args.lr)

    train(args.epochs)
    valid()
    #test()

    # 98.3% was best accuracy in FAST AI
    # Old benchmark 83%
    exit()
