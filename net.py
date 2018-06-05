#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xavier
import chainer
import chainer.links as L


class Encoder(chainer.Chain):

    def __init__(self, x_dim, eps_dim, h_dim=512):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(x_dim, eps_dim, initialW=xavier.Xavier(x_dim, eps_dim))
            self.l2 = L.Linear(eps_dim, h_dim, initialW=xavier.Xavier(eps_dim, h_dim))
            self.l3 = L.Linear(h_dim, eps_dim, initialW=xavier.Xavier(h_dim, eps_dim))

    def __call__(self, x, eps):
        w = self.l1(x)
        h = w + eps
        h = self.l2(h)
        h = self.l3(h)
        return h
