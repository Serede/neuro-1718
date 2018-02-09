#!/usr/bin/env python3

from neuron import Neuron

class MPNeuron(Neuron):
    theta = None

    def __init__(self, name, theta):
        self.theta = theta
        super(MPNeuron, self).__init__(name)

    def __str__(self):
        return 'MP' + super(MPNeuron, self).__str__()

    def f(self, y_in):
        return int(y_in >= self.theta)