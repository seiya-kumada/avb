#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from dataset import *  # noqa
from sampler import *  # noqa
from net import *  # noqa
from chainer import optimizers
from constants import *  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='adversarial variational bayes: avb')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epochs', '-e', default=50, type=int,
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


class UpdateSwitch(object):

    def __init__(self, enc, dec, dis):
        self.encoder = enc
        self.decoder = dec
        self.discriminator = dis

    def update_models(self, enc_updates, dec_updates, dis_updates):
        self.update_model(enc_updates, self.encoder)
        self.update_model(dec_updates, self.decoder)
        self.update_model(dis_updates, self.discriminator)

    def update_model(self, updates, mdl):
        if updates:
            mdl.enable_update()
        else:
            mdl.disable_update()


def setup_optimizer(optimizer, loss_calculator):
    optimizer.setup(loss_calculator)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
    # optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))


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
    encoder = AlternativeEncoder_(x_dim, args.z_dim, args.h_dim)
    decoder = Decoder(args.z_dim, x_dim, args.h_dim)
    discriminator = Discriminator(x_dim, args.z_dim, args.h_dim)

    update_switch = UpdateSwitch(encoder, decoder, discriminator)

    theta_loss_calculator = ThetaLossCalculator_(encoder, decoder)
    phi_loss_calculator = PhiLossCalculator_(encoder, decoder, discriminator)
    psi_loss_calculator = PsiLossCalculator_(encoder, discriminator)

    # _/_/_/ make optimizers

    theta_optimizer = optimizers.Adam()
    setup_optimizer(theta_optimizer, theta_loss_calculator)

    phi_optimizer = optimizers.Adam()
    setup_optimizer(phi_optimizer, phi_loss_calculator)

    psi_optimizer = optimizers.Adam()
    setup_optimizer(psi_optimizer, psi_loss_calculator)

    # _/_/_/ train

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    batches = n_train // args.batch_size
    epoch_theta_losses = []
    epoch_phi_losses = []
    epoch_psi_losses = []
    with chainer.using_config('train', True):
        for epoch in range(args.epochs):
            # shuffle dataset
            sampler.shuffle_xs()

            epoch_theta_loss = 0
            epoch_phi_loss = 0
            epoch_psi_loss = 0

            for i in range(batches):
                xs = sampler.sample_xs()
                zs = sampler.sample_gaussian(0, 1)
                es = sampler.sample_gaussian(0, 1)

                # compute psi-gradient(eq.3.3)
                update_switch.update_models(enc_updates=False, dec_updates=False, dis_updates=True)
                psi_loss = psi_loss_calculator(xs, zs, es)
                update(psi_loss, psi_loss_calculator, psi_optimizer)

                # compute theta-gradient(eq.3.7) in the source paper
                # update_switch.update_models(enc_updates=False, dec_updates=True, dis_updates=False)
                # theta_loss = theta_loss_calculator(xs, zs, es)
                # update(theta_loss, theta_loss_calculator, theta_optimizer)

                # compute phi-gradient(eq.3.7)
                update_switch.update_models(enc_updates=True, dec_updates=True, dis_updates=False)
                phi_loss = phi_loss_calculator(xs, zs, es)
                update(phi_loss, phi_loss_calculator, phi_optimizer)

                # epoch_theta_loss += theta_loss
                epoch_phi_loss += phi_loss
                epoch_psi_loss += psi_loss

            # see loss per epoch
            # epoch_theta_loss /= batches
            epoch_phi_loss /= batches
            epoch_psi_loss /= batches
            print('epoch:{}, theta_loss:{}, phi_loss:{}, psi_loss:{}'.format(epoch,
                                                                             0,
                                                                             epoch_phi_loss.data,
                                                                             epoch_psi_loss.data))
            # epoch_theta_losses.append(epoch_theta_loss.data)
            epoch_phi_losses.append(epoch_phi_loss.data)
            epoch_psi_losses.append(epoch_psi_loss.data)

        # np.save(os.path.join(args.out, 'epoch_theta_losses.npy'), np.array(epoch_theta_losses))
        np.save(os.path.join(args.out, 'epoch_phi_losses.npy'), np.array(epoch_phi_losses))
        np.save(os.path.join(args.out, 'epoch_psi_losses.npy'), np.array(epoch_psi_losses))

        chainer.serializers.save_npz(os.path.join(args.out, 'encoder.npz'), encoder, compression=True)
        chainer.serializers.save_npz(os.path.join(args.out, 'decoder.npz'), decoder, compression=True)
        chainer.serializers.save_npz(os.path.join(args.out, 'discriminator.npz'), discriminator, compression=True)
