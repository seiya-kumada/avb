#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
# TODO: remove init_scope!


class PhiLossCalculator_1(chainer.Chain):

    def __init__(self, theta_loss_calculator, discriminator):
        super(PhiLossCalculator_1, self).__init__()
        with self.init_scope():
            self.theta_loss_calculator = theta_loss_calculator
            self.discriminator = discriminator

    def __call__(self, xs, zs):
        batch_size = xs.shape[0]
        t_loss = F.sum(self.discriminator(xs, zs)) / batch_size
        theta_loss = self.theta_loss_calculator(xs, zs)
        return t_loss + theta_loss


class PhiLossCalculator_2(chainer.Chain):

    def __init__(self, encoder, decoder, discriminator):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.discriminator = discriminator

    def __call__(self, xs, zs, es):
        batch_size = xs.shape[0]
        encoded_zs = self.encoder(xs, es)
        ys = self.decoder(encoded_zs)
        d_loss = F.bernoulli_nll(xs, ys) / batch_size
        t_loss = F.sum(self.discriminator(xs, encoded_zs)) / batch_size
        return t_loss + d_loss, encoded_zs


if __name__ == '__main__':
    import unittest
    import numpy as np
    from decoder import *  # noqa

    class TestPhiLossCalculator_1(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            z = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            decoder = Decoder_1(z_dim, x_dim)
            p = decoder(z)
            self.assertTrue(p.shape == (batch_size, x_dim))
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            theta_loss_calculator = ThetaLossCalculator_1(decoder)
            discriminator = Discriminator_1(x_dim, z_dim)
            phi_loss_calculator = PhiLossCalculator_1(theta_loss_calculator, discriminator)
            loss = phi_loss_calculator(xs, zs)
            self.assertTrue(loss.shape == ())

    unittest.main()
