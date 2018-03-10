#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from network import Network
from copy import deepcopy


class PerNetwork(Network):
    input_length = None
    output_length = None
    theta = None
    learn = None
    synapses = None
    bias = None

    def __init__(self, name, input_length, output_length, theta, learn):
        super(PerNetwork, self).__init__(name)
        self.input_length = input_length
        self.output_length = output_length
        self.theta = theta
        if not 0 < learn <= 1:
            raise ValueError('Learning rate must be within 0 and 1')
        self.learn = learn
        self.synapses = np.zeros(shape=(self.output_length, self.input_length))
        self.bias = np.zeros(shape=(self.output_length,))

    def __str__(self):
        head = super(PerNetwork, self).__str__()
        body = ''
        return '\n'.join([head, body])

    def classify_i(self, datain):
        y_in = np.dot(self.synapses, datain) + self.bias
        y = np.zeros(shape=y_in.shape, dtype=int)
        y[y_in < -self.theta] = -1
        y[y_in > self.theta] = 1
        return y

    def classify(self, datain):
        return np.vstack(tuple(map(lambda x: self.classify_i(x), datain)))

    def train_i(self, datain, dataout, verbose=False):
        y = self.classify_i(datain)
        if verbose:
            print("EX:", dataout, "GOT:", y)
            print("Weights Before", self.synapses)
        errors = (y != dataout)

        updated = False
        for i in range(y.shape[0]):
            if errors[i]:
                delta_synapses = self.learn * dataout[i] * datain
                delta_bias = self.learn * dataout[i]

                self.synapses[i] += delta_synapses
                self.bias[i] += delta_bias

                if (delta_bias != 0).any() or (delta_synapses != 0).any():
                    updated = True

        if verbose:
            print("Weights After", self.synapses)
            print("Updated:", updated)
            print("*"*10)
        return updated

    def train(self, datain, dataout, verbose=False):
        updated = True
        # When during a epoch at least one instance updates weights:
        while updated:
            updated = False
            CIONTADRO = 0
            for input, output in zip(datain, dataout):
                if verbose:
                    print(CIONTADRO)
                    CIONTADRO += 1
                row_updated = self.train_i(input, output, verbose=verbose)
                updated = updated or row_updated
            if verbose:
                print("***" * 10, self.score(datain, dataout))

        return

    def score(self, test_in, test_out):
        res = self.classify(test_in)
        score = (test_out == res).sum() / test_out.size
        return score

    def run(self, datain, verbose=False):
        dataout = []
        for i in range(len(datain)):
            y = self.classify(datain[i])
            dataout.append(' '.join([str(x) for x in y]))
        return dataout
