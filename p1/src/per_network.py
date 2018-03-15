#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from src.dataset import btp, ptb
from src.network import Network

__default_max_epoch__ = 10


class PerNetwork(Network):
    input_length = None
    output_length = None
    theta = None
    learn = None
    synapses = None
    bias = None

    def __init__(self, name, input_length, output_length, theta, learn_rate):
        super(PerNetwork, self).__init__(name)
        self.input_length = input_length
        self.output_length = output_length
        self.theta = theta

        if not 0 < learn_rate <= 1:
            raise ValueError('Learning rate must be within 0 and 1')

        self.learn = learn_rate
        self.synapses = np.zeros(shape=(self.output_length, self.input_length))
        self.bias = np.zeros(shape=(self.output_length,))

    def __str__(self):
        head = super(PerNetwork, self).__str__()
        body = ''
        return '\n'.join([head, body])

    def classify_i_p(self, datain):
        """
        The output is polar
        :param datain:
        :return:
        """
        y_in = np.dot(self.synapses, datain) + self.bias
        y = np.zeros(shape=y_in.shape, dtype=int)

        y[y_in < -self.theta] = -1
        y[y_in > self.theta] = 1

        return y

    def classify(self, datain, binary_output=True):
        polar = np.vstack(tuple(map(lambda x: self.classify_i_p(x), datain)))
        if binary_output:
            return ptb(polar)
        else:
            return polar

    def train(self, datain, dataout, max_epoch=__default_max_epoch__):
        updated = True
        epoch = 0
        dataout_polar = btp(dataout)

        # When during a epoch at least one instance updates weights:
        while updated and epoch < max_epoch:
            updated = False
            for input, output in zip(datain, dataout_polar):
                row_updated = self.train_i_p(input, output)
                updated = updated or row_updated

            epoch += 1

        return

    def train_i_p(self, datain, dataout_polar):
        """
        Expects the output in polar
        :param datain:
        :param dataout_polar:
        :return:
        """
        y = self.classify_i_p(datain)

        errors = (y != dataout_polar)

        delta_synapses = self.learn * np.dot(dataout_polar.reshape((1, -1)).T, datain.reshape((1, -1)))
        delta_bias = self.learn * dataout_polar

        self.synapses[errors] += delta_synapses[errors]
        self.bias[errors] += delta_bias[errors]

        if (delta_bias != 0).any() or (delta_synapses != 0).any():
            updated = True

        return updated

    def score(self, datain, dataout):

        dataout_binary = ptb(dataout)

        res = self.classify(datain, binary_output=True)
        score = (dataout_binary == res).sum() / dataout_binary.size
        return score

    def run(self, datain, binary_output=True):
        return self.classify(datain, binary_output)
