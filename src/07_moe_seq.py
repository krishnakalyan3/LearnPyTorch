#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(train):
    x = train[:, -54]
    y = train[:, 54]


    print(y)
    #return train, val, test

if __name__ == '__main__':
    forest_cvo_type = np.genfromtxt('../data/covtype/covtype.csv', delimiter=',')
    print(forest_cvo_type.shape)
    pass








# Train 100k
# Validation 10k
# Test 50k
# Get good value of gamma and C on 10k