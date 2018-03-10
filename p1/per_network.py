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
            raise ValueError("Learn rate must be between 0 and 1")

        self.learn = learn

        self.synapses = np.zeros(shape=(self.output_length, self.input_length))
        self.bias = np.zeros(shape=(self.output_length,))

    def __str__(self):
        head = super(PerNetwork, self).__str__()
        body = ''
        return '\n'.join([head, body])

    def train(self, train_input, train_output, verbose=False):

        updated = True

        while updated:

            updated = False

            for input,output in zip(train_input,train_output):
                updated = updated or self.__train_one__(input, output)

            print("***" * 10, self.score(train_input, train_output))

        return

    def __train_one__(self, train_input, train_output):

        output = self.__eval_one__(train_input)
        print(output,"->" ,train_output)

        errors = (output != train_output)

        updated = False

        for i in range(output.shape[0]):
            if errors[i]:

                delta_synapses = self.learn * train_output[i] * train_input
                delta_bias = self.learn * train_output[i]

                self.synapses[i] = self.synapses[i] + delta_synapses
                self.bias[i] = self.bias[i] + delta_bias

                if (delta_bias != 0).any() or (delta_synapses != 0).any():
                    updated = True

        return updated

    def eval(self, input):
        return np.vstack(tuple(map(lambda instance: self.__eval_one__(instance), input)))

    def __eval_one__(self, input):

        y_in = np.dot(self.synapses, input) + self.bias

        y = np.zeros(shape=y_in.shape, dtype=int)

        y[y_in < -self.theta] = -1
        y[y_in > self.theta] = 1

        return y

    def score(self, test_input, test_output):
        output = self.eval(test_input)

        score = (test_output == output).sum() / test_output.size

        return score

    def run(self, datain, verbose=False):
        dataout = []
        for i in range(len(datain)):
            y = eval(datain[i])
            dataout.append(' '.join([str(x) for x in y]))
        return dataout
