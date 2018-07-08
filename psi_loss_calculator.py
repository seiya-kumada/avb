#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from constants import *  # noqa
import numpy as np
xp = np
if GPU >= 0:
    xp = chainer.cuda.cupy


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
        super(PsiLossCalculator_3, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.discriminator = discriminator

    def __call__(self, xs, zs, es):
        # batch_size = xs.shape[0]
        encoded_zs = self.encoder(xs, es)
        posterior = self.discriminator(xs, encoded_zs)
        prior = self.discriminator(xs, zs)
        a = F.sigmoid_cross_entropy(posterior, xp.ones_like(posterior).astype(xp.int32))
        b = F.sigmoid_cross_entropy(prior, xp.zeros_like(prior).astype(xp.int32))
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
            zs = xp.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(xp.float32)
            xs = xp.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(xp.float32)
            discriminator = Discriminator_1(x_dim, z_dim)
            if GPU >= 0:
                discriminator.to_gpu()
            # for child in discriminator.children():
            #     print(isinstance(child, chainer.Link))
            rs = discriminator(xs, zs)
            self.assertTrue(rs.shape == (batch_size,))

            psi_loss_calculator = PsiLossCalculator_1(discriminator)
            if GPU >= 0:
                psi_loss_calculator.to_gpu()
            loss = psi_loss_calculator(xs, zs, zs)
            self.assertTrue(loss.shape == ())

    unittest.main()
