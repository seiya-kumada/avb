#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.stats as stats
import numpy as np
import chainer
from constants import *  # noqa


def update_links(net, updates):
    for child in net.children():
        if isinstance(child, chainer.Link):
            if updates:
                child.enable_update()
            else:
                child.disable_update()


def calculate_kl_divergence(zs, es):
    if GPU >= 0:
        zs = chainer.cuda.to_cpu(zs)
        es = chainer.cuda.to_cpu(es)

    v = 0
    for (z, e) in zip(zs, es):
        a = np.prod(stats.norm.pdf(e))
        b = np.prod(stats.norm.pdf(z))
        v += np.log(a / b)
    v /= es.shape[0]
    return v
