#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import argparse
import os
import numpy as np
from utils import *  # noqa
from constants import *  # noqa
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
xp = np
if GPU >= 0:
    xp = chainer.cuda.cupy


def show_arg(name, arg):
    print('{}: {}'.format(name, arg))


def show_args():
    print('> argument:')
    show_arg(' gpu', GPU)
    show_arg(' out', OUT)
    show_arg(' epochs', EPOCHS)
    show_arg(' z_dim', Z_DIM)
    show_arg(' h_dim', H_DIM)
    show_arg(' batch_size', BATCH_SIZE)
    show_arg(' pixel_size', PIXEL_SIZE)
    show_arg(' enc_path', ENC_PATH)
    show_arg(' dec_path', DEC_PATH)
    show_arg(' dis_path', DIS_PATH)
    show_arg(' phi_path', PHI_PATH)
    show_arg(' psi_path', PSI_PATH)


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


def load_if_exists(path, name, model):
    if path:
        print('load trained {}'.format(name))
        chainer.serializers.load_npz(path, model, strict=True)


if __name__ == '__main__':

    # _/_/_/ load arguments

    show_args()

    # _/_/_/ make a dataset

    dataset = Dataset(SAMPLE_SIZE, PIXEL_SIZE)
    dataset.make()
    dataset.split(DATASET_RATIO)
    print('> dataset size:')
    print(' train.shape:\t{}'.format(dataset.train.shape))
    print(' test.shape:\t{}'.format(dataset.test.shape))
    n_train, x_dim = dataset.train.shape

    # _/_/_/ make sampler

    sampler = Sampler(dataset.train, Z_DIM, BATCH_SIZE)

    # _/_/_/ load model

    assert(x_dim == PIXEL_SIZE)
    encoder = Encoder_2(x_dim, Z_DIM, H_DIM)
    decoder = Decoder_1(Z_DIM, x_dim, H_DIM)
    discriminator = Discriminator_1(x_dim, Z_DIM, H_DIM)
    if GPU >= 0:
        encoder.to_gpu()
        decoder.to_gpu()
        discriminator.to_gpu()

    update_switch = UpdateSwitch(encoder, decoder, discriminator)

    phi_loss_calculator = PhiLossCalculator_2(encoder, decoder, discriminator)
    psi_loss_calculator = PsiLossCalculator_3(encoder, discriminator)
    if GPU >= 0:
        phi_loss_calculator.to_gpu()
        psi_loss_calculator.to_gpu()

    # _/_/_/ make optimizers

    phi_optimizer = optimizers.Adam(beta1=BETA1)
    setup_optimizer(phi_optimizer, phi_loss_calculator)

    psi_optimizer = optimizers.Adam(beta1=BETA1)
    setup_optimizer(psi_optimizer, psi_loss_calculator)

    # _/_/_/ if there exist trained models, load them

    load_if_exists(ENC_PATH, 'encoder', encoder)
    load_if_exists(DEC_PATH, 'decoder', decoder)
    load_if_exists(DIS_PATH, 'discriminator', discriminator)
    load_if_exists(PHI_PATH, 'phi_optimizer', phi_optimizer)
    load_if_exists(PSI_PATH, 'psi_optimizer', psi_optimizer)
    load_if_exists(PHI_LOSS_PATH, 'phi_loss_calculator', phi_loss_calculator)
    load_if_exists(PSI_LOSS_PATH, 'psi_loss_calculator', psi_loss_calculator)

    # _/_/_/ train

    if not os.path.exists(OUT):
        os.mkdir(OUT)

    batches = n_train // BATCH_SIZE
    epoch_phi_losses = []
    epoch_psi_losses = []
    epoch_kls = []
    for epoch in range(EPOCHS):
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
                kl = calculate_kl_divergence(encoded_zs.data, es)
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
    np.save(os.path.join(OUT, 'epoch_phi_losses.npy'), np.array(epoch_phi_losses))
    np.save(os.path.join(OUT, 'epoch_psi_losses.npy'), np.array(epoch_psi_losses))
    np.save(os.path.join(OUT, 'epoch_kls.npy'), np.array(epoch_kls))

    chainer.serializers.save_npz(os.path.join(OUT, 'encoder.npz'), encoder, compression=True)
    chainer.serializers.save_npz(os.path.join(OUT, 'decoder.npz'), decoder, compression=True)
    chainer.serializers.save_npz(os.path.join(OUT, 'discriminator.npz'), discriminator, compression=True)
    chainer.serializers.save_npz(os.path.join(OUT, 'phi_loss_calculator.npz'), phi_loss_calculator, compression=True)
    chainer.serializers.save_npz(os.path.join(OUT, 'psi_loss_calculator.npz'), psi_loss_calculator, compression=True)
    chainer.serializers.save_npz(os.path.join(OUT, 'phi_optimizer.npz'), phi_optimizer, compression=True)
    chainer.serializers.save_npz(os.path.join(OUT, 'psi_optimizer.npz'), psi_optimizer, compression=True)
