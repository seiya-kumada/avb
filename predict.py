#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import chainer
import numpy as np
from dataset import *  # noqa
from encoder import *  # noqa
from decoder import *  # noqa
from constants import *  # noqa

SAMPLE_SIZE = 40000


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


def make_xs(index, values):
    indices = np.array([index] * SAMPLE_SIZE)
    return values[indices]


if __name__ == '__main__':
    # load arguments
    args = parse_args()

    # load a model
    x_dim = 4
    encoder = Encoder_2(x_dim, args.z_dim, args.h_dim)
    encoder_path = os.path.join(args.in_dir, 'encoder.npz')
    chainer.serializers.load_npz(encoder_path, encoder, strict=True)

    # sample using gaussian distribution
    gaussian = np.random.normal(0, 1, (SAMPLE_SIZE, args.z_dim)).astype(np.float32)

    # make xs
    dataset = Dataset(SAMPLE_SIZE)
    dataset.make()
    xs = dataset.dataset

    with chainer.using_config('train', False):
        # encode them
        zs = encoder(xs, gaussian)

    z0s = []
    z1s = []
    z2s = []
    z3s = []
    for x, z in zip(xs, zs):
        index = np.argmax(x)
        if index == 0:
            z0s.append(z.data)
        elif index == 1:
            z1s.append(z.data)
        elif index == 2:
            z2s.append(z.data)
        else:
            z3s.append(z.data)

    z0s = np.array(z0s)
    z1s = np.array(z1s)
    z2s = np.array(z2s)
    z3s = np.array(z3s)

    # save them
    np.save(os.path.join(args.in_dir, 'posterior_z0s.npy'), z0s)
    np.save(os.path.join(args.in_dir, 'posterior_z1s.npy'), z1s)
    np.save(os.path.join(args.in_dir, 'posterior_z2s.npy'), z2s)
    np.save(os.path.join(args.in_dir, 'posterior_z3s.npy'), z3s)

    # reconstruct
    decoder = Decoder_1(args.z_dim, x_dim, args.h_dim)
    decoder_path = os.path.join(args.in_dir, 'decoder.npz')
    chainer.serializers.load_npz(decoder_path, decoder, strict=True)

    with chainer.using_config('train', False):
        # encode them
        x0s = decoder(z0s, is_sigmoid=True)
        x1s = decoder(z1s, is_sigmoid=True)
        x2s = decoder(z2s, is_sigmoid=True)
        x3s = decoder(z3s, is_sigmoid=True)

    # save them
    np.save(os.path.join(args.in_dir, 'reconstructed_x0s.npy'), x0s.data)
    np.save(os.path.join(args.in_dir, 'reconstructed_x1s.npy'), x1s.data)
    np.save(os.path.join(args.in_dir, 'reconstructed_x2s.npy'), x2s.data)
    np.save(os.path.join(args.in_dir, 'reconstructed_x3s.npy'), x3s.data)
