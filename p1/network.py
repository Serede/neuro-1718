#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from copy import deepcopy

class Network():
    name = None
    input_layer = None
    output_layer = None
    hidden_layers = None
    synapses = None
    time = None

    def __init__(self, name, input_layer, output_layer, hidden_layers, synapses):
        self.name = name
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.synapses = defaultdict(list)
        for syn in synapses:
            self.synapses[syn.target].append(syn)
        self.time = 0

    def __str__(self):
        header = 'NETWORK \'%s\'\n' % self.name
        return header + ('=' * (len(header)-1)) + '\nLayers:\n' + '\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.hidden_layers]) + '\nSynapses:\n' + '\n'.join([str(s) for s in sum(self.synapses.values(), [])])

    def updateInputLayer(self, values):
        if len(self.input_layer) != len(values):
            raise ValueError('Number of input neurons mismatch in values {}.'.format(values))
        for val, neuron in zip(values, self.input_layer.neurons):
            neuron.f(val)

    def step(self):
        syn = {k: deepcopy(v) for k, v in self.synapses.items()}
        for target in self.synapses.keys():
            target.f(sum([s.weight * s.origin.value for s in syn[target]]))

    def run(self, datain, verbose=False):
        dataout = []
        self.updateInputLayer(datain[0])
        for i in range(1, len(datain) + len(self.hidden_layers)):
            self.step()
            if(i < len(datain)):
                self.updateInputLayer(datain[i])
            else:
                self.updateInputLayer([0] * len(self.input_layer))
            self.time += 1
            if verbose:
                print('Status at time %d:' % self.time)
                print('\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.hidden_layers]))
            dataout.append(' '.join([str(n.value) for n in self.output_layer.neurons]))
        return dataout

    __repr__ = __str__