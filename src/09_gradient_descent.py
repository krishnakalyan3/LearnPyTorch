#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch
dtype = torch.FloatTensor


def mse_loss(a, b, x, y):
    return mse(lin(a,b,x), y)


def mse(y_hat, y):
    return ((y_hat - y) ** 2).mean()


def lin(a,b,x):
    return a*x+b


def gen_fake_data(n, a, b):
    x = np.random.uniform(0,1,n)
    y = lin(a,b,x) + 0.1 * np.random.normal(0,3,n)
    return x, y





if __name__ == '__main__':
    x, y = gen_fake_data(10000, 3., 8.)

    a = Variable(torch.from_numpy(np.random.randn(1)), requires_grad=True)
    b = Variable(torch.from_numpy(np.random.randn(1)), requires_grad=True)

    learning_rate = 1e-3
    for t in range(10000):
        # Forward pass: compute predicted y using operations on Variables
        loss = mse_loss(a, b, x, y)
        if t % 1000 == 0: print(loss.data[0])
