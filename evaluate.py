#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os

EPSILON = 1.0e-6


def parse_args():
    parser = argparse.ArgumentParser(description='adversarial variational bayes: avb')
    parser.add_argument('--in_dir', '-i', default='result',
                        help='Directory to the trained model')
    parser.add_argument('--z_dim', '-z', default=2, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--h_dim', '-hd', default=256, type=int,
                        help='dimention of hidden layer')
    args = parser.parse_args()
    return args


def calculate_cross_entropy(rxs, txs):
    r = txs * np.log(rxs + EPSILON) + (1 - txs) * np.log(1 - rxs)
    r = np.sum(r, axis=1)
    return -np.sum(r)


def make_true_samples(index, rxs):
    size, _ = rxs.shape
    a = np.eye(4)[index].reshape(1, -1)
    return a.repeat(size, axis=0)


def calculate_reconstruction_error(args):
    # load reconstruced samples
    reconstucted_x0s = np.load(os.path.join(args.in_dir, 'reconstructed_x0s.npy'))
    reconstucted_x1s = np.load(os.path.join(args.in_dir, 'reconstructed_x1s.npy'))
    reconstucted_x2s = np.load(os.path.join(args.in_dir, 'reconstructed_x2s.npy'))
    reconstucted_x3s = np.load(os.path.join(args.in_dir, 'reconstructed_x3s.npy'))

    # make true samples
    true_x0s = make_true_samples(0, reconstucted_x0s)
    true_x1s = make_true_samples(1, reconstucted_x1s)
    true_x2s = make_true_samples(2, reconstucted_x2s)
    true_x3s = make_true_samples(3, reconstucted_x3s)

    # calculate reconstruction error
    cross_entropy_0 = calculate_cross_entropy(reconstucted_x0s, true_x0s)
    cross_entropy_1 = calculate_cross_entropy(reconstucted_x1s, true_x1s)
    cross_entropy_2 = calculate_cross_entropy(reconstucted_x2s, true_x2s)
    cross_entropy_3 = calculate_cross_entropy(reconstucted_x3s, true_x3s)

    v = cross_entropy_0 + cross_entropy_1 + cross_entropy_2 + cross_entropy_3
    t = true_x0s.shape[0] + true_x1s.shape[0] + true_x2s.shape[0] + true_x3s.shape[0]
    v /= t
    return v


if __name__ == '__main__':
    args = parse_args()

    reconstruction_error = calculate_reconstruction_error(args)
    print('reconstruction_error: {}'.format(reconstruction_error))
