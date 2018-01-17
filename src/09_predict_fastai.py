#!/usr/bin/env python3
from torch.autograd import Variable
from models import resnext_50
from torch import nn
from other.layers import AdaptiveConcatPool2d, Flatten
import torch
import torchvision.datasets as dsets
from other.utils import load_model
import torchvision.transforms as transforms
from PIL import Image


def layer0():
    return [AdaptiveConcatPool2d(), Flatten()]


def layer1():
    model = [nn.BatchNorm1d(num_features=4096)]
    model.append(nn.Dropout(p=0.25))
    model.append(nn.Linear(in_features=4096, out_features=512))
    model.append(nn.ReLU())
    return model


def layer2():
    model = [nn.BatchNorm1d(num_features=512)]
    model.append(nn.Dropout(p=0.5))
    model.append(nn.Linear(in_features=512, out_features=12))
    model.append(nn.LogSoftmax())
    return model


def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]


def build_model():
    model = resnext_50.resnext_50_32x4d()
    model = cut_model(model, 8)
    model = model + layer0() + layer1() + layer2()
    model = nn.Sequential(*model)
    return model

imsize = 250
preprocess = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])


def image_loader(image_name):
    model.eval()
    image = Image.open(image_name)
    image = preprocess(image).unsqueeze(0)
    image = Variable(image)
    val = model(image)
    predicted = torch.max(val, 1)
    return predicted[1].data.numpy()



if __name__ == '__main__':
    model = build_model()
    model = load_model(model, '../data/models/resnext_50_all_data.h5')
    res = image_loader('/Users/krishnakalyan3/Educational/Plant/data/test/0ad9e7dfb.png')
    print(res)

