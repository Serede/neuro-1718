#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from copy import deepcopy
from network import Network


class PerNetwork(Network):
    input_length = None
    output_length = None
    theta = None
    learn_rate = None

    def __init__(self, name, input_length, output_length, theta, learn_rate):
        super(PerNetwork, self).__init__(name)
        self.input_length = input_length
        self.output_length = output_length
        self.theta = theta
        self.learn_rate = learn_rate
        self.time = 0

    def __str__(self):
        return 'Per-' + super(PerNetwork, self).__str__()

    def run(self, datain, verbose=False):
        # Set weights to zero
        synapses = np.zeros((self.input_length + 1, self.output_length))
        for i in len(datain):
            data = datain[i][:-self.output_length]
            expect = datain[i][-self.output_length:]
            # Evaluate the network for the input
            y_in = np.dot(synapses, np.hstack((1, data)))
            y = np.zeros(y_in.shape, dtype=int)
            y[y_in < -self.theta] = -1
            y[y_in > -self.theta] = 1
            # Adjust weights
            errors = np.where(y != expect)[0]
            if not list(errors):
                break
            for j in errors:
                synapses[j] = synapses[j] + self.learn_rate * expect[j]

