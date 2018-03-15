# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuro.meta.net import Net
import numpy as np


class MPNet(Net):

    def f(self, y_in):
        return (y_in >= self.threshold)

    def train(self, data):
        raise NotImplementedError('McCulloch-Pitts does not support training.')
