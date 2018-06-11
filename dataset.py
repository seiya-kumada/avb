#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1)


class Dataset(object):

    def __init__(self, sample_size, pixel_size=4):
        self.sample_size = sample_size
        self.pixel_size = pixel_size

    # test ok
    def make(self):
        values = np.identity(self.pixel_size).astype(np.float32)
        indices = np.random.randint(0, self.pixel_size, (self.sample_size,))
        self.dataset = values[indices]

    # test ok
    def split(self, ratio):
        total_size = self.dataset.shape[0]
        train_size = int(ratio * total_size)
        self.train = self.dataset[:train_size, :].copy()
        self.test = self.dataset[train_size:, :].copy()

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
            answers = np.array([
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])
            self.assertTrue(np.all(dataset.dataset == answers))
            self.assertTrue(dataset.dataset.dtype == np.float32)
            self.assertTrue(dataset.dataset.shape == (sample_size, pixel_size))

        def test_shift(self):
            sample_size = 5
            pixel_size = 4
            dataset = Dataset(sample_size, pixel_size)
            dataset.make()
            a = dataset.dataset.copy()
            dataset.shift()
            a[a == 0] = -1
            self.assertTrue(np.all(dataset.dataset == a))
            self.assertTrue(dataset.dataset.dtype == np.float32)
            self.assertTrue(dataset.dataset.shape == (sample_size, pixel_size))

        def test_split(self):
            sample_size = 5
            dataset = Dataset(sample_size)
            dataset.make()
            dataset.split(ratio=0.9)
            self.assertTrue(dataset.train.shape == (4, 4))
            self.assertTrue(dataset.test.shape == (1, 4))

    unittest.main()
