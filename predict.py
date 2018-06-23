#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import chainer
import numpy as np
import scipy.stats as stats

from dataset import *  # noqa
from encoder import *  # noqa
from decoder import *  # noqa
from discriminator import *  # noqa
from constants import *  # noqa

SAMPLE_SIZE = 40000
EPSILON = 1.0e-6
np.random.seed(12)


def calculate_cross_entropy(rxs, txs):
    r = txs * np.log(rxs + EPSILON) + (1 - txs) * np.log(1 - rxs)
    r = np.sum(r, axis=1)
    return -np.sum(r)


def make_true_samples(index, rxs):
    size, _ = rxs.shape
    a = np.eye(4)[index].reshape(1, -1)
    return a.repeat(size, axis=0)


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


def encode(encoder, xs, gaussian):
    with chainer.using_config('train', False):
        # encode them
        zs = encoder(xs, gaussian)

    return zs


# この値がマイナスになる。
def calculate_kl_divergence(zs, es):
    v = 0
    for (z, e) in zip(zs, es):
        a = np.prod(stats.norm.pdf(e))
        b = np.prod(stats.norm.pdf(z.data))
        v += np.log(a / b)
    v /= es.shape[0]
    return v


def calculate_kl_divergence_(dis, xs, zs):
    a = 0
    with chainer.using_config('train', False):
        for i in range(zs.shape[0]):
            a += dis(xs, zs)
    a /= zs.shape[0]
    return np.mean(a.data)


def recontruct(decoder, z0s, z1s, z2s, z3s):
    with chainer.using_config('train', False):
        # encode them
        x0s = decoder(z0s, is_sigmoid=True)
        x1s = decoder(z1s, is_sigmoid=True)
        x2s = decoder(z2s, is_sigmoid=True)
        x3s = decoder(z3s, is_sigmoid=True)
    return (x0s, x1s, x2s, x3s)


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

    # _/_/_/ encode

    zs = encode(encoder, xs, gaussian)
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

    # _/_/_/ reconstruct

    decoder = Decoder_1(args.z_dim, x_dim, args.h_dim)
    decoder_path = os.path.join(args.in_dir, 'decoder.npz')
    chainer.serializers.load_npz(decoder_path, decoder, strict=True)

    (x0s, x1s, x2s, x3s) = recontruct(decoder, z0s, z1s, z2s, z3s)

    # save them
    np.save(os.path.join(args.in_dir, 'reconstructed_x0s.npy'), x0s.data)
    np.save(os.path.join(args.in_dir, 'reconstructed_x1s.npy'), x1s.data)
    np.save(os.path.join(args.in_dir, 'reconstructed_x2s.npy'), x2s.data)
    np.save(os.path.join(args.in_dir, 'reconstructed_x3s.npy'), x3s.data)

    # _/_/_/ calculate reconstruction error

    # make true samples
    true_x0s = make_true_samples(0, x0s.data)
    true_x1s = make_true_samples(1, x1s.data)
    true_x2s = make_true_samples(2, x2s.data)
    true_x3s = make_true_samples(3, x3s.data)

    # calculate reconstruction error
    cross_entropy_0 = calculate_cross_entropy(x0s.data, true_x0s)
    cross_entropy_1 = calculate_cross_entropy(x1s.data, true_x1s)
    cross_entropy_2 = calculate_cross_entropy(x2s.data, true_x2s)
    cross_entropy_3 = calculate_cross_entropy(x3s.data, true_x3s)

    v = cross_entropy_0 + cross_entropy_1 + cross_entropy_2 + cross_entropy_3
    t = true_x0s.shape[0] + true_x1s.shape[0] + true_x2s.shape[0] + true_x3s.shape[0]
    v /= t
    print('reconstruction error:{}'.format(v))

    # _/_/_/ calculate KL divergence

    # discriminator = Discriminator_1(x_dim, args.z_dim, args.h_dim)
    # discriminator_path = os.path.join(args.in_dir, 'discriminator.npz')
    # chainer.serializers.load_npz(discriminator_path, discriminator, strict=True)
    # kld = calculate_kl_divergence_(discriminator, xs, zs)
    kld = calculate_kl_divergence(zs, gaussian)
    print('KL divergence:{}'.format(kld))
