#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from theta_loss_calculator import *  # noqa
from discriminator import *  # noqa


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
        super(PhiLossCalculator_2, self).__init__()
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
    from decoder import *  # noqa
    from constants import *  # noqa
    import numpy as np
    xp = np
    if GPU >= 0:
        xp = chainer.cuda.cupy

    class TestPhiLossCalculator_1(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            z = xp.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(xp.float32)

            decoder = Decoder_1(z_dim, x_dim)
            if GPU >= 0:
                decoder.to_gpu()

            p = decoder(z)
            self.assertTrue(p.shape == (batch_size, x_dim))
            xs = xp.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(xp.float32)
            zs = xp.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(xp.float32)

            theta_loss_calculator = ThetaLossCalculator_1(decoder)
            if GPU >= 0:
                theta_loss_calculator.to_gpu()

            discriminator = Discriminator_1(x_dim, z_dim)
            if GPU >= 0:
                discriminator.to_gpu()

            phi_loss_calculator = PhiLossCalculator_1(theta_loss_calculator, discriminator)
            if GPU >= 0:
                phi_loss_calculator.to_gpu()

            loss = phi_loss_calculator(xs, zs)
            self.assertTrue(loss.shape == ())

    unittest.main()
