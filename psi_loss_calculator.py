#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import numpy as np


class PsiLossCalculator_1(chainer.Chain):

    def __init__(self, discriminator):
        super(PsiLossCalculator_1, self).__init__()
        with self.init_scope():
            self.discriminator = discriminator

    def __call__(self, xs, encoded_zs, zs):
        batch_size = xs.shape[0]
        a = F.log(F.sigmoid(self.discriminator(xs, encoded_zs)))
        b = F.log(1.0 - F.sigmoid(self.discriminator(xs, zs)))
        c = -F.sum(a + b)
        return c / batch_size


class PsiLossCalculator_2(chainer.Chain):

    def __init__(self, encoder, discriminator):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.discriminator = discriminator

    def __call__(self, xs, zs, es):
        batch_size = xs.shape[0]
        encoded_zs = self.encoder(xs, es)
        a = F.log(F.sigmoid(self.discriminator(xs, encoded_zs)))
        b = F.log(1.0 - F.sigmoid(self.discriminator(xs, zs)))
        c = -F.sum(a + b)
        return c / batch_size


class PsiLossCalculator_3(chainer.Chain):

    def __init__(self, encoder, discriminator):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.discriminator = discriminator

    def __call__(self, xs, zs, es):
        batch_size = xs.shape[0]
        encoded_zs = self.encoder(xs, es)
        posterior = self.discriminator(xs, encoded_zs)
        prior = self.discriminator(xs, zs)
        a = F.sigmoid_cross_entropy(posterior, np.ones_like(posterior).astype(np.int32))
        b = F.sigmoid_cross_entropy(prior, np.zeros_like(prior).astype(np.int32))
        c = F.sum(a + b)
        return c


if __name__ == '__main__':
    import unittest
    from discriminator import *  # noqa

    class TestPsiLossCalculator(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            discriminator = Discriminator_1(x_dim, z_dim)
            # for child in discriminator.children():
            #     print(isinstance(child, chainer.Link))
            rs = discriminator(xs, zs)
            self.assertTrue(rs.shape == (batch_size,))

            psi_loss_calculator = PsiLossCalculator_1(discriminator)
            loss = psi_loss_calculator(xs, zs, zs)
            self.assertTrue(loss.shape == ())

    unittest.main()
