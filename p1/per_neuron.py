#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import Neuron


class PerNeuron(Neuron):
    theta = None
    value = None

    def __init__(self, name, theta, value=0):
        self.theta = theta
        self.value = value
        super(PerNeuron, self).__init__(name)

    def __str__(self):
        return 'Per' + super(PerNeuron, self).__str__() + '{{th = {}, v = {}}}'.format(self.theta, self.value)

    def f(self, y_in):
        if y_in < -self.theta:
            self.value = -1
        if y_in > self.theta:
            self.value = 1
        else:
            self.value = 0
