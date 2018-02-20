#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from copy import deepcopy

class Network():
    name = None
    input_layer = None
    output_layer = None
    middle_layers = None
    synapses = None
    time = None

    def __init__(self, name, input_layer, output_layer, middle_layers, synapses):
        self.name = name
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.middle_layers = middle_layers
        self.synapses = defaultdict(list)
        for syn in synapses:
            self.synapses[syn.target].append(syn)
        self.time = 0

    def __str__(self):
        header = 'NETWORK \'%s\'\n' % self.name
        return header + ('=' * (len(header)-1)) + '\nLayers:\n' + '\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.middle_layers]) + '\nSynapses:\n' + '\n'.join([str(s) for s in sum(self.synapses.values(), [])])

    def updateInputLayer(self, values):
        if len(self.input_layer.neurons) != len(values):
            raise ValueError('Number of input neurons mismatch in input file.')
        for val, neuron in zip(values, self.input_layer.neurons):
            neuron.f(val)

    def step(self):
        syn = {k: deepcopy(v) for k, v in self.synapses.items()}
        for target in self.synapses.keys():
            target.f(sum([s.weight * s.origin.value for s in syn[target]]))

    def run(self, input_file, time_shift):
        values = [np.asarray(line.split(' ')).astype(float) for line in input_file.read().splitlines()]
        self.updateInputLayer(values[0])
        for i in range(1, time_shift):
            self.step()
            self.updateInputLayer(values[i])
            self.time += 1
            print('Status at time %d:' % self.time)
            print('\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.middle_layers]))

    __repr__ = __str__