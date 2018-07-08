#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import chainer
from constants import *  # noqa
np.random.seed(1)
xp = np
if GPU >= 0:
    xp = chainer.cuda.cupy
    xp.random.seed(1)


class Dataset(object):

    def __init__(self, sample_size, pixel_size=4):
        self.sample_size = sample_size
        self.pixel_size = pixel_size

    # test ok
    def make(self):
        values = xp.identity(self.pixel_size).astype(xp.float32)
        indices = xp.random.randint(0, self.pixel_size, (self.sample_size,))
        self.dataset = values[indices]

    # test ok
    def split(self, ratio):
        total_size = self.dataset.shape[0]
        train_size = int(ratio * total_size)
        self.train = self.dataset[:train_size, :].copy()
        self.test = self.dataset[train_size:, :].copy()

    # test ok
    def shift(self):
        self.dataset = 2 * self.dataset - 1


if __name__ == '__main__':
    import unittest

    class TestDataset(unittest.TestCase):

        def test_make_dataset(self):
            sample_size = 5
            pixel_size = 4
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            if GPU >= 0:
                answers = xp.array([
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0]])
            else:
                answers = xp.array([
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])
            self.assertTrue(xp.all(dataset.dataset == answers))
            self.assertTrue(dataset.dataset.dtype == xp.float32)
            self.assertTrue(dataset.dataset.shape == (sample_size, pixel_size))

        def test_make_dataset_2(self):
            sample_size = 5
            pixel_size = 7
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            if GPU >= 0:
                answers = xp.array([
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0]])
            else:
                answers = xp.array([
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0]])
            self.assertTrue(xp.all(dataset.dataset == answers))
            self.assertTrue(dataset.dataset.dtype == xp.float32)
            self.assertTrue(dataset.dataset.shape == (sample_size, pixel_size))

        def test_shift(self):
            sample_size = 5
            pixel_size = 7
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            a = dataset.dataset.copy()
            dataset.shift()
            a[a == 0] = -1
            self.assertTrue(xp.all(dataset.dataset == a))
            self.assertTrue(dataset.dataset.dtype == xp.float32)
            self.assertTrue(dataset.dataset.shape == (sample_size, pixel_size))

        def test_split(self):
            sample_size = 5
            pixel_size = 8
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            dataset.split(ratio=0.9)
            self.assertTrue(dataset.train.shape == (4, pixel_size))
            self.assertTrue(dataset.test.shape == (1, pixel_size))

    unittest.main()
