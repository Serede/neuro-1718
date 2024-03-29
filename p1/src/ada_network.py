#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from src.dataset import btp, ptb
from src.network import Network

__default_max_epoch__ = 100
__default_threshold__ = 0.01


class AdaNetwork(Network):
    input_size = None
    output_size = None
    learn_rate = None
    bias = None
    synapses = None

    def __init__(self, name, input_size, output_size, learn_rate):
        super(AdaNetwork, self).__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        self.learn_rate = learn_rate
        self.bias = np.random.uniform(-1., 1., output_size)
        self.synapses = np.random.uniform(-1., 1., input_size *
                                          output_size).reshape((output_size, input_size))

    def train(self, datain, dataout, max_epoch=__default_max_epoch__,
              threshold=__default_threshold__):

        dataout_polar = btp(dataout)

        epochs = 0
        delta = np.inf

        while threshold < delta and epochs < max_epoch:
            delta = self.train_all_instances(datain, dataout_polar)
            epochs += 1

        return

    def train_all_instances(self, input_polar, output_polar):

        delta = -np.inf

        for input_i, output_i in zip(input_polar, output_polar):
            delta = max(delta, self.train_i(input_i, output_i))

        return delta

    def train_i(self, input_i, output_i):

        y_in = np.dot(self.synapses, input_i) + self.bias

        delta_w = self.learn_rate * np.dot((output_i - y_in).reshape((self.output_size, 1)),
                                           input_i.reshape((1, self.input_size)))

        delta_b = self.learn_rate * (output_i - y_in)

        self.synapses = self.synapses + delta_w
        self.bias = self.bias + delta_b

        max_delta_w = np.max(np.abs(delta_w))
        max_delta_b = np.max(np.abs(delta_b))
        max_delta = max([max_delta_b, max_delta_w])

        return max_delta

    def classify_i_p(self, datain):
        """
        The output is polar
        :param datain:
        :return:
        """

        y_in = np.dot(self.synapses, datain) + self.bias
        y = np.zeros(shape=y_in.shape, dtype=int)

        y[y_in < 0] = -1
        y[y_in > 0] = 1

        return y

    def classify(self, datain, binary_output=True):
        polar = np.vstack(tuple(map(lambda x: self.classify_i_p(x), datain)))
        if binary_output:
            return ptb(polar)
        else:
            return polar

    def score(self, datain, dataout):
        dataout_binary = ptb(dataout)

        res = self.classify(datain, binary_output=True)
        score = (dataout_binary == res).sum() / dataout_binary.size

        return score

    def run(self, datain, binary_output=True):
        return self.classify(datain, binary_output)
