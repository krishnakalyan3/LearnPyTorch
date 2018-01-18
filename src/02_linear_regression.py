#!/usr/bin/env python3
# Please download the housing dataset from https://www.kaggle.com/apratim87/housingdata

import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.basic_models import LinearRegression
import numpy as np

HOUSING_DATA = '../data/Regression/housingdata.csv'
housing_data = np.genfromtxt(HOUSING_DATA, delimiter=",")
x_train = np.array(housing_data[:, 0:1], dtype=np.float32)
y_train = np.array(housing_data[:, 13:], dtype=np.float32)


def train():
    for epoch in range(args.epochs):
        inputs = Variable(torch.from_numpy(x_train))
        targets = Variable(torch.from_numpy(y_train))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Loss: %.4f'
              % (epoch + 1, args.epochs, loss.data[0]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FeedForward Example')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()

    model = LinearRegression(input_size=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train()

    # Plot the graph
    predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.show()
