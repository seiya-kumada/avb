#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F


class ThetaLossCalculator_1(chainer.Chain):

    def __init__(self, decoder):
        super(ThetaLossCalculator_1, self).__init__()
        with self.init_scope():
            self.decoder = decoder

    def __call__(self, x, z):
        batch_size = x.shape[0]
        y = self.decoder(z)
        loss = F.bernoulli_nll(x, y)
        return loss / batch_size


class ThetaLossCalculator_2(chainer.Chain):

    def __init__(self, encoder, decoder):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder

    def __call__(self, xs, zs, es):
        batch_size = xs.shape[0]
        encoded_zs = self.encoder(xs, es)
        ys = self.decoder(encoded_zs)
        loss = F.bernoulli_nll(xs, ys)
        return loss / batch_size


if __name__ == '__main__':
    import unittest
    import numpy as np
    from decoder import *  # noqa

    class TestThetaLossCalculator_1(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            decoder = Decoder_1(z_dim, x_dim)
            ps = decoder(zs)
            self.assertTrue(ps.shape == (batch_size, x_dim))
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            theta_loss_calculator = ThetaLossCalculator_1(decoder)
            loss = theta_loss_calculator(xs, zs)
            self.assertTrue(loss.shape == ())

    unittest.main()
