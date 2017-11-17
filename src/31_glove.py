#!/usr/bin/env python3
import argparse
import numpy as np
from other.utils import cpu_stats
from torch.autograd import Variable
import torch
import torch.optim as optim

PATH = '../data/short_story.txt'
CONTEXT_SIZE = 1
EMBED_SIZE = 2
XMAX = 2
ALPHA = 0.75


def read_data():
    data = []
    with open(PATH, 'r', encoding='UTF-8') as f:
        words = f.read().split(' ')
        for word in words:
            data.append(word.strip())
    return data


def co_occ_matrix(word, vocab):
    # Word index mapping
    mapping = {word: index for index, word in enumerate(vocab)}

    len_word_list = len(word)
    len_vocab = len(vocab)
    print(len_vocab)
    co_mat = np.zeros((len_vocab, len_vocab))

    for i in range(len_word_list):
        for j in range(1, CONTEXT_SIZE + 1):
            index = mapping[word[i]]
            if i - j > 0:
                lind = mapping[word_list[i - j]]
                co_mat[index, lind] += 1.0 / j
            if i + j < len_word_list:
                rind = mapping[word_list[i + j]]
                co_mat[index, rind] += 1.0 / j

    return co_mat


def wf(x):
    if x < XMAX:
        return (x / XMAX) ** ALPHA
    return 1


def gen_batch(comat, coocs):
    sample = np.random.choice(np.arange(len(coocs)), size=args.batch_size, replace=False)
    l_vecs, r_vecs, covals, l_v_bias, r_v_bias = [], [], [], [], []
    for chosen in sample:
        ind = tuple(coocs[chosen])
        l_vecs.append(l_embed[ind[0]])
        r_vecs.append(r_embed[ind[1]])
        covals.append(comat[ind])
        l_v_bias.append(l_biases[ind[0]])
        r_v_bias.append(r_biases[ind[1]])
    return l_vecs, r_vecs, covals, l_v_bias, r_v_bias


if __name__ == '__main__':
    # print(cpu_stats())
    parser = argparse.ArgumentParser(description='PyTorch FeedForward Example')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--gpu', default=False, help='CUDA enable')
    args = parser.parse_args()

    word_list = read_data()
    vocab_list = np.unique(word_list)
    len_vocab = len(vocab_list)
    len_word = len(word_list)

    co_mat = co_occ_matrix(word_list, vocab_list)

    l_embed, r_embed = [
        [Variable(torch.from_numpy(np.random.normal(0, 0.01, (EMBED_SIZE, 1))),
                  requires_grad=True) for j in range(len_vocab)] for i in range(2)]
    l_biases, r_biases = [
        [Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)),
                  requires_grad=True) for j in range(len_vocab)] for i in range(2)]

    coocs = np.transpose(np.nonzero(co_mat))
    batch = gen_batch(co_mat, coocs)

    optimizer = optim.Adam(l_embed + r_embed + l_biases + r_biases, lr=args.lr)

    for epoch in range(args.epochs):
        num_batches = int(len_word / args.batch_size)
        avg_loss = 0.0
        for batch in range(num_batches):
            optimizer.zero_grad()
            l_vecs, r_vecs, covals, l_v_bias, r_v_bias = gen_batch(co_mat, coocs)
            loss = sum([torch.mul((torch.dot(l_vecs[i], r_vecs[i]) +
                                   l_v_bias[i] + r_v_bias[i] - np.log(covals[i])) ** 2,
                                   wf(covals[i])) for i in range(args.batch_size)])
            avg_loss += loss.data[0] / num_batches
            loss.backward()
            optimizer.step()
        print("Average loss for epoch " + str(epoch + 1) + ": ", avg_loss)
