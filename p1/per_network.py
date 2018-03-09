#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from network import Network


class PerNetwork(Network):
    input_length = None
    output_length = None
    theta = None
    learn_rate = None
    synapses = None

    def __init__(self, name, input_length, output_length, theta, learn_rate):
        super(PerNetwork, self).__init__(name)
        self.input_length = input_length
        self.output_length = output_length
        self.theta = theta
        self.learn_rate = learn_rate
        self.synapses = np.zeros((self.output_length, self.input_length + 1))

    def __str__(self):
        return 'Per-' + super(PerNetwork, self).__str__()

    def train(self, datain, verbose=False):
        # Set weights to zero
        self.synapses.fill(0)
        for i in range(len(datain)):
            data = datain[i][:-self.output_length]
            expect = datain[i][-self.output_length:]
            # Evaluate the network for the input
            print(self.synapses, data, expect)
            y_in = np.dot(self.synapses, np.hstack((1, data)))
            y = np.zeros(y_in.shape, dtype=int)
            y[y_in < -self.theta] = -1
            y[y_in > self.theta] = 1
            # Adjust weights
            errors = np.where(y != expect)[0]
            print('errors = {}'.format(errors))
            # if not list(errors):
            #    break
            for j in errors:
                print(expect[j])
                self.synapses[j] = self.synapses[j] + \
                    self.learn_rate * expect[j]
                print('Updating {} with value {}.'.format(
                    self.synapses[j], self.learn_rate * expect[j]))

    def run(self, datain, verbose=False):
        dataout = []
        for i in range(len(datain)):
            data = datain[i][:-self.output_length]
            # Evaluate the network for the input
            y_in = np.dot(self.synapses, np.hstack((1, data)))
            y = np.zeros(y_in.shape, dtype=int)
            y[y_in < -self.theta] = -1
            y[y_in > self.theta] = 1
            dataout.append(' '.join([str(x) for x in y]))
        return dataout
