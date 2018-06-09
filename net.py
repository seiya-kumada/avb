#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xavier
import chainer
import chainer.links as L
import chainer.functions as F

# what does 'reuse' in tensorflow mean?
# https://qiita.com/halhorn/items/6805b1fd3f8ff74840df


# This class represents q(z|x) or z(x,e).
class Encoder(chainer.Chain):

    def __init__(self, x_dim, eps_dim, h_dim=512):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.a1 = L.Linear(eps_dim, x_dim, initialW=xavier.Xavier(eps_dim, x_dim))
            self.a2 = L.Linear(eps_dim, h_dim, initialW=xavier.Xavier(eps_dim, h_dim))
            self.a3 = L.Linear(eps_dim, h_dim, initialW=xavier.Xavier(eps_dim, h_dim))

            self.l1 = L.Linear(x_dim, h_dim, initialW=xavier.Xavier(eps_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, eps_dim, initialW=chainer.initializers.Normal(scale=1e-5))

    def merge_1(self, x, eps):
        h = self.a1(eps)
        return h + x

    def merge_2(self, x, eps):
        h = self.a2(eps)
        return h + x

    def merge_3(self, x, eps):
        h = self.a3(eps)
        return h + x

    def __call__(self, x, eps):
        h = self.merge_1(x, eps)
        h = self.l1(h)
        h = F.softplus(h)

        h = self.merge_2(h, eps)
        h = self.l2(h)
        h = F.softplus(h)

        h = self.merge_3(h, eps)
        h = self.l3(h)
        return h


# This class represents a paramter of p(x|z).
# Now a paramter of the Bernoulli distribution is calculated.
class Decoder(chainer.Chain):

    def __init__(self, z_dim, x_dim, h_dim=512):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(z_dim, h_dim, initialW=xavier.Xavier(z_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, x_dim, initialW=xavier.Xavier(h_dim, x_dim))

    def __call__(self, z):
        h = self.l1(z)
        h = F.softplus(h)

        h = self.l2(h)
        h = F.softplus(h)

        h = self.l3(h)
        return h


# This class represents a dicriminator.
class Discriminator(chainer.Chain):

    def __init__(self, x_dim, z_dim, h_dim=512):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.xl1 = L.Linear(x_dim, h_dim, initialW=xavier.Xavier(x_dim, h_dim))
            self.xl2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))
            self.zl1 = L.Linear(z_dim, h_dim, initialW=xavier.Xavier(z_dim, h_dim))
            self.zl2 = L.Linear(h_dim, h_dim, initialW=xavier.Xavier(h_dim, h_dim))

    def __call__(self, x, z):
        hx = self.xl1(x)
        hx = F.softplus(hx)
        hx = self.xl2(hx)
        hx = F.softplus(hx)

        hz = self.zl1(z)
        hz = F.softplus(hz)
        hz = self.zl2(hz)
        hz = F.softplus(hz)

        h = F.sum(hx * hz, axis=1)
        return h


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestEncoder(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            eps_dim = 2
            x = chainer.Variable(np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32))
            eps = chainer.Variable(np.arange(batch_size * eps_dim).reshape(batch_size, eps_dim).astype(np.float32))
            encoder = Encoder(x_dim, eps_dim)
            z = encoder(x, eps)
            self.assertTrue(z.shape == (batch_size, eps_dim))

    class TestDecoder(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            z = chainer.Variable(np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32))
            x = chainer.Variable(np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32))
            decoder = Decoder(z_dim, x_dim)
            x = decoder(z)
            self.assertTrue(x.shape == (batch_size, x_dim))

    class TestDiscriminator(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            z = chainer.Variable(np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32))
            x = chainer.Variable(np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32))
            discriminator = Discriminator(x_dim, z_dim)
            r = discriminator(x, z)
            self.assertTrue(r.shape == (batch_size,))

    unittest.main()
