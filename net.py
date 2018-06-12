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


# This class is an alternative encoder.
class AlternativeEncoder(chainer.Chain):

    def __init__(self, x_dim, eps_dim, h_dim=512):
        super(AlternativeEncoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(x_dim + eps_dim, h_dim)  # , initialW=xavier.Xavier(x_dim + eps_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))
            self.l4 = L.Linear(h_dim, eps_dim)  # , initialW=xavier.Xavier(h_dim, eps_dim))

    def __call__(self, x, eps):
        h = F.concat((x, eps), axis=1)

        h = self.l1(h)
        h = F.relu(h)

        h = self.l2(h)
        h = F.relu(h)

        h = self.l3(h)
        h = F.relu(h)

        h = self.l4(h)
        return h


# This class represents a parameter of p(x|z).
# Now a paramter of the Bernoulli distribution is calculated.
class Decoder(chainer.Chain):

    def __init__(self, z_dim, x_dim=1, h_dim=512):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(z_dim, h_dim)  # , initialW=xavier.Xavier(z_dim, h_dim))
            self.l2 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))
            self.l3 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))
            self.l4 = L.Linear(h_dim, x_dim)  # , initialW=xavier.Xavier(h_dim, x_dim))

    def __call__(self, z):
        h = self.l1(z)
        h = F.relu(h)

        h = self.l2(h)
        h = F.relu(h)

        h = self.l3(h)
        h = F.relu(h)

        h = self.l4(h)
        return h


class ThetaLossCalculator(chainer.Chain):

    def __init__(self, decoder):
        super(ThetaLossCalculator, self).__init__()
        with self.init_scope():
            self.decoder = decoder

    def __call__(self, x, z):
        batch_size = x.shape[0]
        y = self.decoder(z)
        loss = F.bernoulli_nll(x, y)
        return loss / batch_size


# This class represents a dicriminator.
class Discriminator(chainer.Chain):

    def __init__(self, x_dim, z_dim, h_dim=512):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.xl1 = L.Linear(x_dim, h_dim)  # , initialW=xavier.Xavier(x_dim, h_dim))
            self.xl2 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))
            self.xl3 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))
            self.zl1 = L.Linear(z_dim, h_dim)  # , initialW=xavier.Xavier(z_dim, h_dim))
            self.zl2 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))
            self.zl3 = L.Linear(h_dim, h_dim)  # , initialW=xavier.Xavier(h_dim, h_dim))

    def __call__(self, xs, zs):
        xs = 2 * xs - 1
        hx = self.xl1(xs)
        hx = F.relu(hx)
        hx = self.xl2(hx)
        hx = F.relu(hx)
        hx = self.xl3(hx)
        hx = F.relu(hx)

        hz = self.zl1(zs)
        hz = F.relu(hz)
        hz = self.zl2(hz)
        hz = F.relu(hz)
        hz = self.zl3(hz)
        hz = F.relu(hz)
        h = F.sum(hx * hz, axis=1)
        return h


class PhiLossCalculator(chainer.Chain):

    def __init__(self, theta_loss_calculator, discriminator):
        super(PhiLossCalculator, self).__init__()
        with self.init_scope():
            self.theta_loss_calculator = theta_loss_calculator
            self.discriminator = discriminator

    def __call__(self, xs, zs):
        batch_size = xs.shape[0]
        t_loss = F.sum(self.discriminator(xs, zs)) / batch_size
        theta_loss = self.theta_loss_calculator(xs, zs)
        return t_loss + theta_loss


class PsiLossCalculator(chainer.Chain):

    def __init__(self, discriminator):
        super(PsiLossCalculator, self).__init__()
        with self.init_scope():
            self.discriminator = discriminator

    def __call__(self, xs, encoded_zs, zs):
        batch_size = xs.shape[0]
        a = F.log(F.sigmoid(self.discriminator(xs, encoded_zs)))
        b = F.log(1.0 - F.sigmoid(self.discriminator(xs, zs)))
        c = -F.sum(a + b)
        return c / batch_size


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestEncoder(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            eps_dim = 2
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            eps = np.arange(batch_size * eps_dim).reshape(batch_size, eps_dim).astype(np.float32)
            encoder = Encoder(x_dim, eps_dim)
            zs = encoder(xs, eps)
            self.assertTrue(zs.shape == (batch_size, eps_dim))

    class TestAlternativeEncoder(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            eps_dim = 2
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            eps = np.arange(batch_size * eps_dim).reshape(batch_size, eps_dim).astype(np.float32)
            encoder = AlternativeEncoder(x_dim, eps_dim)
            zs = encoder(xs, eps)
            self.assertTrue(zs.shape == (batch_size, eps_dim))

    class TestDecoder(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            decoder = Decoder(z_dim, x_dim)
            xs = decoder(zs)
            self.assertTrue(xs.shape == (batch_size, x_dim))

    class TestThetaLossCalculator(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            decoder = Decoder(z_dim, x_dim)
            ps = decoder(zs)
            self.assertTrue(ps.shape == (batch_size, x_dim))
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            theta_loss_calculator = ThetaLossCalculator(decoder)
            loss = theta_loss_calculator(xs, zs)
            self.assertTrue(loss.shape == ())

    class TestDiscriminator(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            discriminator = Discriminator(x_dim, z_dim)
            rs = discriminator(xs, zs)
            self.assertTrue(rs.shape == (batch_size,))

        def test_average(self):
            x = np.array([[1, 2], [2, 1]]).astype(np.float32)
            y = np.array([[3, 4], [4, 3]]).astype(np.float32)
            print(x)
            print(y)
            z = x * y
            print(z)
            w = F.average(F.sum(z, axis=1))
            print(w.data)

    class TestPhiLossCalculator(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            z = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            decoder = Decoder(z_dim, x_dim)
            p = decoder(z)
            self.assertTrue(p.shape == (batch_size, x_dim))
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            theta_loss_calculator = ThetaLossCalculator(decoder)
            discriminator = Discriminator(x_dim, z_dim)
            phi_loss_calculator = PhiLossCalculator(theta_loss_calculator, discriminator)
            loss = phi_loss_calculator(xs, zs)
            self.assertTrue(loss.shape == ())

    class TestPsiLossCalculator(unittest.TestCase):

        def test_call(self):
            batch_size = 3
            x_dim = 4
            z_dim = 2
            zs = np.arange(batch_size * z_dim).reshape(batch_size, z_dim).astype(np.float32)
            xs = np.arange(batch_size * x_dim).reshape(batch_size, x_dim).astype(np.float32)
            discriminator = Discriminator(x_dim, z_dim)
            rs = discriminator(xs, zs)
            self.assertTrue(rs.shape == (batch_size,))

            psi_loss_calculator = PsiLossCalculator(discriminator)
            loss = psi_loss_calculator(xs, zs, zs)
            self.assertTrue(loss.shape == ())

    unittest.main()
