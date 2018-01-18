#!/usr/bin/env python3
import torch
from torch.autograd import Variable
import torch.nn as nn

# Example 1
a = torch.IntTensor([1, 2, 3])
b = torch.IntTensor([4, 5, 6])
m = a * b
print(a.numpy())
print(m.numpy())

# Example 2
x = Variable(torch.Tensor([6]), requires_grad=False)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w * x + b
y.backward()

print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1

# Example 3
x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
pred = linear(x)

# Compute loss and back prop
loss = criterion(pred, y)
print('loss: ', loss.data[0])
loss.backward()

# 1 step optimization
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)
optimizer.step()


pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.data[0])
