#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from predict import *  # noqa
from dataset import *  # noqa
from sampler import *  # noqa
from encoder import *  # noqa
from decoder import *  # noqa
from discriminator import *  # noqa
from phi_loss_calculator import *  # noqa
from psi_loss_calculator import *  # noqa
from chainer import optimizers
from constants import *  # noqa
# http://studylog.hateblo.jp/entry/2016/01/05/212830
# http://ensekitt.hatenablog.com/entry/2017/12/13/200000


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
    parser.add_argument('--h_dim', '-hd', default=256, type=int,
                        help='dimention of hidden layer')
    parser.add_argument('--batch_size', '-b', default=500, type=int,
                        help='learning minibatch size')
    parser.add_argument('--enc_path', '-encp', type=str, default='result_200/encoder.npz',
                        help='path to a trained encoder model')
    parser.add_argument('--dec_path', '-decp', type=str, default='result_200/decoder.npz',
                        help='path to a trained decoder model')
    parser.add_argument('--dis_path', '-disp', type=str, default='result_200/discriminator.npz',
                        help='path to a trained discriminator model')
    parser.add_argument('--phi_path', '-phip', type=str, default='result_200/phi_optimizer.npz',
                        help='path to a trained phi optimizer')
    parser.add_argument('--psi_path', '-psip', type=str, default='result_200/psi_optimizer.npz',
                        help='path to a trained psi optimizer')
    parser.add_argument('--phi_loss_path', '-philp', type=str, default='result_200/phi_loss_calculator.npz',
                        help='path to a trained phi loss calculator')
    parser.add_argument('--psi_loss_path', '-psilp', type=str, default='result_200/psi_loss_calculator.npz',
                        help='path to a trained psi loss calculator')
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
    show_arg(' enc_path', args.enc_path)
    show_arg(' dec_path', args.dec_path)
    show_arg(' dis_path', args.dis_path)
    show_arg(' phi_path', args.phi_path)
    show_arg(' psi_path', args.psi_path)


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


def evaluate(xs, encoder, discriminator):
    size, _ = xs.shape
    es = np.random.normal(0, 1, (size, 2)).astype(np.float32)
    zs = np.random.normal(0, 1, (size, 2)).astype(np.float32)

    with chainer.using_config('train', False):
        encoded_zs = encoder(xs, es)
        posterior = np.mean(discriminator(xs, encoded_zs).data)
        prior = np.mean(discriminator(xs, zs).data)
    return posterior, prior


def load_if_exists(path, name, model):
    if path:
        print('load trained {}'.format(name))
        chainer.serializers.load_npz(path, model, strict=True)


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
    encoder = Encoder_2(x_dim, args.z_dim, args.h_dim)
    decoder = Decoder_1(args.z_dim, x_dim, args.h_dim)
    discriminator = Discriminator_1(x_dim, args.z_dim, args.h_dim)

    update_switch = UpdateSwitch(encoder, decoder, discriminator)

    phi_loss_calculator = PhiLossCalculator_2(encoder, decoder, discriminator)
    psi_loss_calculator = PsiLossCalculator_3(encoder, discriminator)

    # _/_/_/ make optimizers

    beta1 = 0.4
    phi_optimizer = optimizers.Adam(beta1=beta1)
    setup_optimizer(phi_optimizer, phi_loss_calculator)

    psi_optimizer = optimizers.Adam(beta1=beta1)
    setup_optimizer(psi_optimizer, psi_loss_calculator)

    # _/_/_/ if there exist trained models, load them

    load_if_exists(args.enc_path, 'encoder', encoder)
    load_if_exists(args.dec_path, 'decoder', decoder)
    load_if_exists(args.dis_path, 'discriminator', discriminator)
    load_if_exists(args.phi_path, 'phi_optimizer', phi_optimizer)
    load_if_exists(args.psi_path, 'psi_optimizer', psi_optimizer)
    load_if_exists(args.phi_loss_path, 'phi_loss_calculator', phi_loss_calculator)
    load_if_exists(args.psi_loss_path, 'psi_loss_calculator', psi_loss_calculator)

    # _/_/_/ train

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    batches = n_train // args.batch_size
    epoch_phi_losses = []
    epoch_psi_losses = []
    epoch_kls = []
    for epoch in range(args.epochs):
        with chainer.using_config('train', True):
            # shuffle dataset
            sampler.shuffle_xs()

            epoch_phi_loss = 0
            epoch_psi_loss = 0
            epoch_kl = 0

            for i in range(batches):
                xs = sampler.sample_xs()
                zs = sampler.sample_zs()
                es = sampler.sample_es()

                # compute psi-gradient(eq.3.3)
                update_switch.update_models(enc_updates=False, dec_updates=False, dis_updates=True)
                psi_loss = psi_loss_calculator(xs, zs, es)
                update(psi_loss, psi_loss_calculator, psi_optimizer)
                epoch_psi_loss += psi_loss

                # compute phi-gradient(eq.3.7)
                update_switch.update_models(enc_updates=True, dec_updates=True, dis_updates=False)
                phi_loss, encoded_zs = phi_loss_calculator(xs, zs, es)
                kl = calculate_kl_divergence(encoded_zs, es)
                update(phi_loss, phi_loss_calculator, phi_optimizer)
                epoch_phi_loss += phi_loss
                epoch_kl += kl
            # end for ...
            # see loss per epoch
            epoch_phi_loss /= batches
            epoch_psi_loss /= batches
            epoch_kl /= batches
        # end with ...
        print('epoch:{}, phi_loss:{}, psi_loss:{}, kl:{}'.format(epoch, epoch_phi_loss.data,
                                                                 epoch_psi_loss.data, epoch_kl))
        epoch_phi_losses.append(epoch_phi_loss.data)
        epoch_psi_losses.append(epoch_psi_loss.data)
        epoch_kls.append(epoch_kl)
    # end for ...
    np.save(os.path.join(args.out, 'epoch_phi_losses.npy'), np.array(epoch_phi_losses))
    np.save(os.path.join(args.out, 'epoch_psi_losses.npy'), np.array(epoch_psi_losses))
    np.save(os.path.join(args.out, 'epoch_kls.npy'), np.array(epoch_kls))

    chainer.serializers.save_npz(os.path.join(args.out, 'encoder.npz'), encoder, compression=True)
    chainer.serializers.save_npz(os.path.join(args.out, 'decoder.npz'), decoder, compression=True)
    chainer.serializers.save_npz(os.path.join(args.out, 'discriminator.npz'), discriminator, compression=True)
    chainer.serializers.save_npz(os.path.join(args.out, 'phi_loss_calculator.npz'), phi_loss_calculator, compression=True)
    chainer.serializers.save_npz(os.path.join(args.out, 'psi_loss_calculator.npz'), psi_loss_calculator, compression=True)
    chainer.serializers.save_npz(os.path.join(args.out, 'phi_optimizer.npz'), phi_optimizer, compression=True)
    chainer.serializers.save_npz(os.path.join(args.out, 'psi_optimizer.npz'), psi_optimizer, compression=True)
