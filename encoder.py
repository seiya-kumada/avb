#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xavier
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from utils import *  # noqa

# what does 'reuse' in tensorflow mean?
# https://qiita.com/halhorn/items/6805b1fd3f8ff74840df
# https://gist.github.com/poolio/b71eb943d6537d01f46e7b20e9225149


# This class represents q(z|x) or z(x,e).
class Encoder_1(chainer.Chain):

    def __init__(self, x_dim, eps_dim, h_dim=512):
        super(Encoder_1, self).__init__()
        with self.init_scope():
            self.a1 = L.Linear(eps_dim, x_dim, initialW=xavier.Xavier(eps_dim, x_dim))
            self.a2 = L.Linear(eps_dim, h_dim, initialW=xavier.Xavier(eps_dim, h_dim))
            self.a3 = L.Linear(eps_dim, h_dim, initialW=xavier.Xavier(eps_dim, h_dim))

            self.l1 = L.Linear(x_dim, h_dim, initialW=xavier.Xavier(eps_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, eps_dim, initialW=chainer.initializers.Normal(scale=1e-5))

    def update(self, updates):
        update_links(self, updates)

    def merge_1(self, x, eps):
        h = self.a1(eps)
        return h + x

    def merge_2(self, x, eps):
        h = self.a2(eps)
        return h + x

    def merge_3(self, x, eps):
        h = self.a3(eps)
        return h + x

    def __call__(self, xs, es, activation=F.softplus):
        xs = 2 * xs - 1
        h = self.merge_1(xs, es)
        h = self.l1(h)
        h = activation(h)

        h = self.merge_2(h, es)
        h = self.l2(h)
        h = activation(h)

        h = self.merge_3(h, es)
        h = self.l3(h)
        return h


# This class is an alternative encoder.
class Encoder_2(chainer.Chain):

    def __init__(self, x_dim, eps_dim, h_dim=512):
        super(Encoder_2, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(x_dim + eps_dim, h_dim, initialW=xavier.Xavier(x_dim + eps_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l4 = L.Linear(h_dim, eps_dim, initialW=xavier.Xavier(h_dim, eps_dim))

    def update(self, updates):
        update_links(self, updates)

    def __call__(self, xs, es, activation=F.relu):
        xs = 2 * xs - 1
        h = F.concat((xs, es), axis=1)

        h = self.l1(h)
        h = activation(h)

        h = self.l2(h)
        h = activation(h)

        h = self.l3(h)
        h = activation(h)

        h = self.l4(h)
        return h


class Encoder_3(chainer.Chain):

    def __init__(self, x_dim, eps_dim, h_dim=512):
        super(Encoder_3, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(x_dim, h_dim, initialW=xavier.Xavier(x_dim + eps_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l4 = L.Linear(h_dim, eps_dim, initialW=xavier.Xavier(h_dim, eps_dim))
            self.a1 = L.Linear(eps_dim, x_dim, initialW=xavier.Xavier(eps_dim, x_dim))

    def merge_1(self, x, eps):
        h = self.a1(eps)
        return h + x

    def update(self, updates):
        update_links(self, updates)

    def __call__(self, xs, es, activation=F.relu):
        xs = 2 * xs - 1
        h = self.merge_1(xs, es)

        h = self.l1(h)
        h = activation(h)

        h = self.l2(h)
        h = activation(h)

        h = self.l3(h)
        h = activation(h)

        h = self.l4(h)
        return h


class Encoder_4(chainer.Chain):

    def __init__(self, x_dim, eps_dim, h_dim=512):
        super(Encoder_4, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(x_dim + eps_dim, h_dim, initialW=xavier.Xavier(x_dim + eps_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, eps_dim, initialW=xavier.Xavier(h_dim, eps_dim))

    def update(self, updates):
        update_links(self, updates)

    def __call__(self, xs, es, activation=F.relu):
        h = F.concat((xs, es), axis=1)

        h = self.l1(h)
        h = activation(h)

        h = self.l2(h)
        h = activation(h)

        h = self.l3(h)
        return h


if __name__ == '__main__':
    import unittest

    class TestEncoder_1(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            eps_dim = 2
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            eps = np.arange(batch_size * eps_dim).reshape(batch_size, eps_dim).astype(np.float32)
            encoder = Encoder_1(x_dim, eps_dim)
            zs = encoder(xs, eps)
            self.assertTrue(zs.shape == (batch_size, eps_dim))

    class TestEncoder_2(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            eps_dim = 2
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            eps = np.arange(batch_size * eps_dim).reshape(batch_size, eps_dim).astype(np.float32)
            encoder = Encoder_2(x_dim, eps_dim)
            zs = encoder(xs, eps)
            self.assertTrue(zs.shape == (batch_size, eps_dim))

    class TestEncoder_4(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            eps_dim = 2
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            eps = np.arange(batch_size * eps_dim).reshape(batch_size, eps_dim).astype(np.float32)
            encoder = Encoder_4(x_dim, eps_dim)
            zs = encoder(xs, eps)
            self.assertTrue(zs.shape == (batch_size, eps_dim))

    unittest.main()
