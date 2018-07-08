#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import argparse
import os
import chainer
import collections
import numpy as np
import scipy.stats as stats
from constants import *  # noqa
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


def make_true_samples(index, rxs, pixel_size):
    size, _ = rxs.shape
    a = np.eye(pixel_size)[index].reshape(1, -1)
    return a.repeat(size, axis=0)


def make_xs(index, values):
    indices = np.array([index] * SAMPLE_SIZE)
    return values[indices]


def encode(encoder, xs, gaussian):
    with chainer.using_config('train', False):
        # encode them
        zs = encoder(xs, gaussian)

    return zs


# この値がマイナスになる...
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


def recontruct(decoder, zs):
    xs = collections.defaultdict(list)
    with chainer.using_config('train', False):
        # encode them
        for (k, v) in zs.items():
            xs[k] = decoder(zs[k], is_sigmoid=True)
    return xs


if __name__ == '__main__':

    # load a model
    x_dim = PIXEL_SIZE
    encoder = Encoder_2(x_dim, Z_DIM, H_DIM)
    encoder_path = os.path.join(OUT, 'encoder.npz')
    chainer.serializers.load_npz(encoder_path, encoder, strict=True)

    # sample using gaussian distribution
    gaussian = np.random.normal(0, 1, (SAMPLE_SIZE, Z_DIM)).astype(np.float32)

    # make xs
    dataset = Dataset(SAMPLE_SIZE, PIXEL_SIZE)
    dataset.make()
    xs = dataset.dataset

    # _/_/_/ encode

    zs = encode(encoder, xs, gaussian)
    zis = collections.defaultdict(list)
    for x, z in zip(xs, zs):
        index = np.argmax(x)
        zis[index].append(z.data)

    for (k, v) in zis.items():
        zis[k] = np.array(v)

    # save them
    for (k, v) in zis.items():
        np.save(os.path.join(OUT, 'posterior_z{}s.npy'.format(k)), zis[k])

    # _/_/_/ reconstruct

    decoder = Decoder_1(Z_DIM, x_dim, H_DIM)
    decoder_path = os.path.join(OUT, 'decoder.npz')
    chainer.serializers.load_npz(decoder_path, decoder, strict=True)

    rec_xis = recontruct(decoder, zis)

    # save them
    for (k, v) in rec_xis.items():
        np.save(os.path.join(OUT, 'reconstructed_x{}s.npy'.format(k)), rec_xis[k].data)

    # _/_/_/ calculate reconstruction error

    # make true samples
    true_xis = collections.defaultdict(list)
    for (k, v) in rec_xis.items():
        true_xis[k] = make_true_samples(k, rec_xis[k].data, len(rec_xis))

    # calculate reconstruction error
    cross_entropies = collections.defaultdict(list)
    for (k, v) in true_xis.items():
        cross_entropies[k] = calculate_cross_entropy(rec_xis[k].data, true_xis[k])

    v = sum(cross_entropies.values())
    t = sum([a.shape[0] for (_, a) in true_xis.items()])
    v /= t
    print('reconstruction error:{}'.format(v))

    # _/_/_/ calculate KL divergence

    # discriminator = Discriminator_1(x_dim, args.z_dim, args.h_dim)
    # discriminator_path = os.path.join(args.in_dir, 'discriminator.npz')
    # chainer.serializers.load_npz(discriminator_path, discriminator, strict=True)
    # kld = calculate_kl_divergence_(discriminator, xs, zs)
    kld = calculate_kl_divergence(zs, gaussian)
    print('KL divergence:{}'.format(kld))
