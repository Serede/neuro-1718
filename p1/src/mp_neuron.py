#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.neuron import Neuron


class MPNeuron(Neuron):
    theta = None
    value_next = None
    value = None
    input = False

    def __init__(self, name, theta, value=0, value_next=0, input=False):
        super(MPNeuron, self).__init__(name)
        self.theta = theta
        self.value = value
        self.value_next = value_next
        self.input = input

    def f(self, y_in):
        if self.input:
            self.value_next = self.value = int(y_in >= self.theta)
        else:
            self.value_next = self.value
            self.value = int(y_in >= self.theta)
