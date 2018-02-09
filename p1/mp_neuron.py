#!/usr/bin/env python3

from neuron import Neuron

class MPNeuron(Neuron):
    theta = None

    def __init__(self, theta):
        self.theta = theta

    def f(self, y_in):
        return int(y_in >= self.theta)