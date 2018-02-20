#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import Neuron

class MPNeuron(Neuron):
    theta = None
    value = None

    def __init__(self, name, theta, value=0):
        self.theta = theta
        self.value = value
        super(MPNeuron, self).__init__(name)

    def __str__(self):
        return 'MP' + super(MPNeuron, self).__str__() + '{{th = {}, v = {}}}'.format(self.theta, self.value)

    def f(self, y_in):
        self.value = int(y_in >= self.theta)
        print('Updated {} to {} ({} >= {})'.format(self, self.value, y_in, self.theta))