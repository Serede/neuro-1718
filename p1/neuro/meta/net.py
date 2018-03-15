# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np

"""Input shape index"""
_in_ = 0

"""Output shape index"""
_out_ = -1


class Net(ABC):
    name = None
    shape = None
    threshold = None
    weights = None

    def __init__(self, name, shape, threshold):
        self.name = name
        self.shape = shape
        self.threshold = threshold
        self.weights = np.zeros((sum(shape), sum(shape)))

    def __str__(self):
        pass

    __repr__ = __str__

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def f(self, y_in):
        pass

    def test(self, data):
        res = []
        y = np.zeros(sum(self.shape))
        if len(self.shape) > 3:
            z = np.zeros((len(self.shape)-3, self.shape[_in_]))
            data = np.vstack((data, z))
        print('data', data, '\n')
        for d in data:
            t = y[self.shape[_in_]:]
            y = np.hstack((np.squeeze(np.asarray(x)) for x in (d, t)))
            y_in = np.dot(self.weights, y).T
            y = np.vectorize(self.f)(y_in).astype(int)
            y = (y_in >= self.threshold).astype(int)
            res.append(y[-self.shape[_out_]:].T.tolist())
        return res
