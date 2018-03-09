#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuron import Neuron


class MPNeuron(Neuron):
    theta = None
    value = None

    def __init__(self, name, theta, value=0):
        super(MPNeuron, self).__init__(name)
        self.theta = theta
        self.value = value

    def f(self, y_in):
        self.value = int(y_in >= self.theta)
