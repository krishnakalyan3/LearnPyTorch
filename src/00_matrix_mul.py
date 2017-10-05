#!/usr/bin/env python3

import torch

a = torch.IntTensor([1, 2, 3])
b = torch.IntTensor([4, 5, 6])
m = a * b
print(a.numpy())
print(m.numpy())