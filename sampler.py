#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1)


class Sampler(object):

    # test ok
    def __init__(self, dataset, z_dim, batch_size):
        self.xs = dataset
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.xs_size, _ = dataset.shape
        self.indices = np.arange(self.xs_size)

    # test ok
    def shuffle_xs(self):
        np.random.shuffle(self.indices)

    # test ok
    def sample_xs(self):
        vs = self.xs[self.indices[:self.batch_size]]
        self.indices = np.roll(self.indices, self.batch_size)
        return vs

    # test ok
    def sample_gaussian(self, mean, sigma):
        return np.random.normal(mean, sigma, (self.batch_size, self.z_dim)).astype(np.float32)


if __name__ == '__main__':
    import unittest
    from dataset import *  # noqa
    import collections

    class TestSampler(unittest.TestCase):

        def test_init(self):
            sample_size = 50
            pixel_size = 4
            ratio = 0.9
            batch_size = 10
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            dataset.split(ratio=ratio)
            self.assertTrue((int(sample_size * ratio), pixel_size) == dataset.train.shape)
            sampler = Sampler(dataset.train, z_dim=4, batch_size=batch_size)
            self.assertTrue(np.all(sampler.xs == dataset.train))
            self.assertTrue(sampler.batch_size == batch_size)
            self.assertTrue(sampler.xs_size == 45)
            self.assertTrue(np.all(sampler.indices == np.arange(45)))

        def test_shuffle(self):
            sample_size = 100
            pixel_size = 4
            ratio = 0.1
            batch_size = 2
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            dataset.split(ratio=ratio)
            self.assertTrue((int(sample_size * ratio), pixel_size) == dataset.train.shape)
            sampler = Sampler(dataset.train, z_dim=4, batch_size=batch_size)
            self.assertTrue(np.all(sampler.xs == dataset.train))
            self.assertTrue(sampler.batch_size == batch_size)
            a = sampler.indices.copy()
            sampler.shuffle_xs()
            b = sampler.indices
            self.assertFalse(np.all(a == b))
            self.assertTrue(np.sum(a) == np.sum(b))

        def test_sample_xs(self):
            sample_size = 10
            pixel_size = 4
            ratio = 0.9
            z_dim = 2
            batch_size = 3
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            dataset.split(ratio=ratio)
            self.assertTrue((int(sample_size * ratio), pixel_size) == dataset.train.shape)
            sampler = Sampler(dataset.train, z_dim, batch_size=batch_size)
            b0 = collections.defaultdict(int)
            for r in sampler.xs:
                r = tuple(r.tolist())
                b0[r] += 1

            batches = 3
            b1 = collections.defaultdict(int)
            for i in range(batches):
                xs = sampler.sample_xs()
                for r in xs:
                    r = tuple(r.tolist())
                    b1[r] += 1

            self.assertTrue(len(b0) == len(b1))
            for k in b0.keys():
                self.assertTrue(b1[k] == b0[k])

        def test_sample_gaussian(self):
            sample_size = 50
            pixel_size = 4
            ratio = 0.9
            z_dim = 2
            batch_size = 10
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            dataset.split(ratio=ratio)
            self.assertTrue((int(sample_size * ratio), pixel_size) == dataset.train.shape)
            sampler = Sampler(dataset.train, z_dim, batch_size=batch_size)
            zs = sampler.sample_gaussian(0, 1)
            self.assertTrue((batch_size, z_dim) == zs.shape)
            self.assertTrue(zs.dtype == np.float32)

    unittest.main()
