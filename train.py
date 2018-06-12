#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from dataset import *  # noqa
from sampler import *  # noqa
from net import *  # noqa
from chainer import optimizers

DATASET_RATIO = 0.9
SAMPLE_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser(description='adversarial variational bayes: avb')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epochs', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--z_dim', '-z', default=2, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--h_dim', '-hd', default=512, type=int,
                        help='dimention of hidden layer')
    parser.add_argument('--batch_size', '-b', default=10, type=int,
                        help='learning minibatch size')
    args = parser.parse_args()
    return args


def show_arg(name, arg):
    print('{}: {}'.format(name, arg))


def show_args(args):
    print('> argument:')
    show_arg(' gpu', args.gpu)
    show_arg(' out', args.out)
    show_arg(' epochs', args.epochs)
    show_arg(' z_dim', args.z_dim)
    show_arg(' h_dim', args.h_dim)
    show_arg(' batch_size', args.batch_size)


def update(loss, calculator, optimizer):
    calculator.cleargrads()
    loss.backward()
    optimizer.update()


if __name__ == '__main__':

    # _/_/_/ load arguments

    args = parse_args()
    show_args(args)

    # _/_/_/ make a dataset

    dataset = Dataset(SAMPLE_SIZE)
    dataset.make()
    dataset.split(DATASET_RATIO)
    print('> dataset size:')
    print(' train.shape:\t{}'.format(dataset.train.shape))
    print(' test.shape:\t{}'.format(dataset.test.shape))
    n_train, x_dim = dataset.train.shape

    # _/_/_/ make sampler

    sampler = Sampler(dataset.train, args.z_dim, args.batch_size)

    # _/_/_/ load model

    assert(x_dim == 4)
    encoder = AlternativeEncoder(x_dim, args.z_dim, args.h_dim)
    decoder = Decoder(args.z_dim, x_dim, args.h_dim)
    discriminator = Discriminator(x_dim, args.z_dim, args.h_dim)

    theta_loss_calculator = ThetaLossCalculator(decoder)
    phi_loss_calculator = PhiLossCalculator(theta_loss_calculator, discriminator)
    psi_loss_calculator = PsiLossCalculator(discriminator)

    # _/_/_/ make optimizers

    theta_optimizer = optimizers.Adam()
    theta_optimizer.setup(theta_loss_calculator)

    phi_optimizer = optimizers.Adam()
    phi_optimizer.setup(phi_loss_calculator)

    psi_optimizer = optimizers.Adam()
    psi_optimizer.setup(psi_loss_calculator)

    # _/_/_/ train

    batches = n_train // args.batch_size

    for epoch in range(args.epochs):
        # shuffle dataset
        sampler.shuffle_xs()

        with chainer.using_config('train', True):
            for i in range(batches):
                xs = sampler.sample_xs()
                zs = sampler.sample_gaussian(0, 1)
                es = sampler.sample_gaussian(0, 1)

                encoded_zs = encoder(xs, es)

                # compute theta-gradient(eq.3.7) in the source paper
                theta_loss = theta_loss_calculator(xs, encoded_zs)
                update(theta_loss, theta_loss_calculator, theta_optimizer)

                # compute phi-gradient(eq.3.7)
                phi_loss = phi_loss_calculator(xs, encoded_zs)
                update(phi_loss, phi_loss_calculator, phi_optimizer)

                # compute psi-gradient(eq.3.3)
                psi_loss = psi_loss_calculator(xs, encoded_zs, zs)
                update(psi_loss, psi_loss_calculator, psi_optimizer)
        # see loss per epoch
        print('epoch:{}, theta_loss:{}, phi_loss:{}, psi_loss:{}'.format(epoch, theta_loss.data, phi_loss.data, psi_loss.data))
