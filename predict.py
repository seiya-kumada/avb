#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import net
import chainer
import numpy as np
from constants import *  # noqa

SAMPLE_SIZE = 5000


def parse_args():
    parser = argparse.ArgumentParser(description='adversarial variational bayes: avb')
    parser.add_argument('--in_dir', '-i', default='result',
                        help='Directory to the trained model')
    parser.add_argument('--z_dim', '-z', default=2, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--h_dim', '-hd', default=512, type=int,
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
    encoder = net.AlternativeEncoder(x_dim, args.z_dim, args.h_dim)
    encoder_path = os.path.join(args.in_dir, 'encoder.npz')
    chainer.serializers.load_npz(encoder_path, encoder, strict=True)

    # sample using gaussian distribution
    gaussian_0 = np.random.normal(0, 1, (SAMPLE_SIZE, args.z_dim)).astype(np.float32)
    gaussian_1 = np.random.normal(0, 1, (SAMPLE_SIZE, args.z_dim)).astype(np.float32)
    gaussian_2 = np.random.normal(0, 1, (SAMPLE_SIZE, args.z_dim)).astype(np.float32)
    gaussian_3 = np.random.normal(0, 1, (SAMPLE_SIZE, args.z_dim)).astype(np.float32)

    # make xs
    values = np.identity(x_dim).astype(np.float32)
    x0s = make_xs(0, values)
    x1s = make_xs(1, values)
    x2s = make_xs(2, values)
    x3s = make_xs(3, values)

    with chainer.using_config('train', False):
        # encode them
        z0s = encoder(x0s, gaussian_0)
        z1s = encoder(x1s, gaussian_1)
        z2s = encoder(x2s, gaussian_2)
        z3s = encoder(x3s, gaussian_3)

    # save them
    np.save(os.path.join(args.in_dir, 'z0s.npy'), z0s.data)
    np.save(os.path.join(args.in_dir, 'z1s.npy'), z1s.data)
    np.save(os.path.join(args.in_dir, 'z2s.npy'), z2s.data)
    np.save(os.path.join(args.in_dir, 'z3s.npy'), z3s.data)

    # reconstruct
    decoder = net.Decoder(args.z_dim, x_dim, args.h_dim)
    decoder_path = os.path.join(args.in_dir, 'decoder.npz')
    chainer.serializers.load_npz(decoder_path, decoder, strict=True)

    with chainer.using_config('train', False):
        # encode them
        x0s = decoder(z0s, is_sigmoid=True)
        x1s = decoder(z1s, is_sigmoid=True)
        x2s = decoder(z2s, is_sigmoid=True)
        x3s = decoder(z3s, is_sigmoid=True)

    # save them
    np.save(os.path.join(args.in_dir, 'x0s.npy'), x0s.data)
    np.save(os.path.join(args.in_dir, 'x1s.npy'), x1s.data)
    np.save(os.path.join(args.in_dir, 'x2s.npy'), x2s.data)
    np.save(os.path.join(args.in_dir, 'x3s.npy'), x3s.data)
