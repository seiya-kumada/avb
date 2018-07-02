#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xavier
import chainer
import chainer.links as L
import chainer.functions as F
from utils import *  # noqa


# This class represents a parameter of p(x|z).
# A paramter of the Bernoulli distribution is calculated.
class Decoder_1(chainer.Chain):

    def __init__(self, z_dim, x_dim=1, h_dim=512):
        super(Decoder_1, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(z_dim, h_dim, initialW=xavier.Xavier(z_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l4 = L.Linear(h_dim, x_dim, initialW=xavier.Xavier(h_dim, x_dim))

    def update(self, updates):
        update_links(self, updates)

    def __call__(self, zs, activation=F.tanh, is_sigmoid=False):
        h = self.l1(zs)
        h = activation(h)

        h = self.l2(h)
        h = activation(h)

        h = self.l3(h)
        h = activation(h)

        h = self.l4(h)
        if is_sigmoid:
            h = F.sigmoid(h)
        return h


class Decoder_2(chainer.Chain):

    def __init__(self, z_dim, x_dim=1, h_dim=512):
        super(Decoder_2, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(z_dim, h_dim, initialW=xavier.Xavier(z_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, x_dim, initialW=xavier.Xavier(h_dim, x_dim))

    def update(self, updates):
        update_links(self, updates)

    def __call__(self, zs, activation=F.relu):
        h = self.l1(zs)
        h = activation(h)

        h = self.l2(h)
        h = activation(h)

        h = self.l3(h)
        h = F.sigmoid(h)
        return h


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestDecoder_1(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            decoder = Decoder_1(z_dim, x_dim)
            xs = decoder(zs)
            self.assertTrue(xs.shape == (batch_size, x_dim))

    class TestDecoder_2(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            decoder = Decoder_2(z_dim, x_dim)
            xs = decoder(zs)
            self.assertTrue(xs.shape == (batch_size, x_dim))

    unittest.main()
