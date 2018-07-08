#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
import xavier
from utils import *  # noqa


# This class represents a dicriminator.
class Discriminator_1(chainer.Chain):

    def __init__(self, x_dim, z_dim, h_dim=512):
        super(Discriminator_1, self).__init__()
        self.h_dim = h_dim
        with self.init_scope():
            self.xl1 = L.Linear(x_dim, h_dim, initialW=xavier.Xavier(x_dim, h_dim))
            self.xl2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.xl3 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.zl1 = L.Linear(z_dim, h_dim, initialW=xavier.Xavier(z_dim, h_dim))
            self.zl2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.zl3 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))

    def update(self, updates):
        update_links(self, updates)

    def __call__(self, xs, zs, activation=F.relu):
        xs = 2 * xs - 1
        hx = self.xl1(xs)
        hx = activation(hx)
        hx = self.xl2(hx)
        hx = activation(hx)
        hx = self.xl3(hx)
        hx = activation(hx)

        hz = self.zl1(zs)
        hz = activation(hz)
        hz = self.zl2(hz)
        hz = activation(hz)
        hz = self.zl3(hz)
        hz = activation(hz)
        h = F.sum(hx * hz, axis=1) / self.h_dim
        return h


class Discriminator_2(chainer.Chain):

    def __init__(self, x_dim, z_dim, h_dim=512):
        super(Discriminator_2, self).__init__()
        self.h_dim = h_dim
        with self.init_scope():
            self.l1 = L.Linear(x_dim + z_dim, h_dim, initialW=xavier.Xavier(x_dim + z_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, 1, initialW=xavier.Xavier(h_dim, 1))

    def update(self, updates):
        update_links(self, updates)

    def __call__(self, xs, zs, activation=F.relu):
        xs = 2 * xs - 1
        h = F.concat((xs, zs), axis=1)

        h = self.l1(h)
        h = activation(h)

        h = self.l2(h)
        h = activation(h)

        h = self.l3(h)
        return F.squeeze(h)


if __name__ == '__main__':
    import unittest
    import numpy as np
    from constants import *  # noqa
    xp = np
    if GPU >= 0:
        xp = chainer.cuda.cupy

    class TestDiscriminator_1(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = xp.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(xp.float32)
            xs = xp.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(xp.float32)
            discriminator = Discriminator_1(x_dim, z_dim)
            if GPU >= 0:
                discriminator.to_gpu()
            rs = discriminator(xs, zs)
            self.assertTrue(rs.shape == (batch_size,))

    class TestDiscriminator_2(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = xp.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(xp.float32)
            xs = xp.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(xp.float32)
            discriminator = Discriminator_2(x_dim, z_dim)
            if GPU >= 0:
                discriminator.to_gpu()
            rs = discriminator(xs, zs)
            self.assertTrue(rs.shape == (batch_size,))

    unittest.main()
