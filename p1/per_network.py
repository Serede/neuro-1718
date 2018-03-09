#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from network import Network


class PerNetwork(Network):
    input_length = None
    output_length = None
    theta = None
    learn = None
    synapses = None

    def __init__(self, name, input_length, output_length, theta, learn):
        super(PerNetwork, self).__init__(name)
        self.input_length = input_length
        self.output_length = output_length
        self.theta = theta
        self.learn = learn
        self.synapses = np.zeros((self.output_length, self.input_length + 1))

    def __str__(self):
        head = super(PerNetwork, self).__str__()
        body = ''
        return '\n'.join([head, body])

    def eval(self, data):
        y_in = np.dot(self.synapses, np.hstack((1, data)))
        y = np.zeros(y_in.shape, dtype=int)
        y[y_in < -self.theta] = -1
        y[y_in > self.theta] = 1
        return y

    def train(self, datain, dataout, verbose=False):
        for i in range(len(datain)):
            y = eval(datain[i])
            for j in np.where(y != dataout)[0]:
                self.synapses[j] += self.learn * dataout[j]

    def run(self, datain, verbose=False):
        dataout = []
        for i in range(len(datain)):
            y = eval(datain[i])
            dataout.append(' '.join([str(x) for x in y]))
        return dataout
