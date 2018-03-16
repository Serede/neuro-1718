# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Abstract neural network module.

Attributes:
    _i (int): Input layer index.
    _o (int): Output layer index.
"""

from abc import ABC, abstractmethod
import numpy as np
import itertools

_i = 0
_o = -1


class Net(ABC):
    """Abstract neural network class.

    Attributes:
        name (str): Net instance name.
        layers (tuple): Layer sizes (first is input, last is output).
        threshold (str): Net-wide cell activation threshold.
    """

    name = None
    layers = None
    threshold = None

    _n = None
    _synapses = None

    def __init__(self, name, layers, threshold):
        self.name = name
        self.layers = layers
        self.threshold = threshold
        self._n = sum(layers) + 1
        self._synapses = np.zeros(shape=(self._n, self._n))

    def __str__(self):
        pass

    __repr__ = __str__

    @abstractmethod
    def f(self, y_in):
        pass

    @abstractmethod
    def train(self, data):
        pass

    def test(self, data):
        res = []
        y = np.zeros(self._n)
        if len(self.layers) > 3:
            z = np.zeros((len(self.layers)-3, self.layers[_i]))
            data = np.vstack((data, z))
        print('data', data, '\_n')
        for d in data:
            t = y[self.layers[_i]:]
            y = np.hstack((np.squeeze(np.asarray(x)) for x in (d, t)))
            y_in = np.dot(self._synapses, y).T
            y = np.vectorize(self.f)(y_in).astype(int)
            y = (y_in >= self.threshold).astype(int)
            res.append(y[-self.layers[_o]:].T.tolist())
        return res
