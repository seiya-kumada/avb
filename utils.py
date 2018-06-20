#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer


def update_links(net, updates):
    for child in net.children():
        if isinstance(child, chainer.Link):
            if updates:
                child.enable_update()
            else:
                child.disable_update()
