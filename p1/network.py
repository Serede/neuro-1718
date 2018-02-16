#!/usr/bin/env python3

import numpy as np

class Network():
    name = None
    layers = None
    synapses = None
    time = None

    def __init__(self, name, input_layer, output_layer, middle_layers, synapses):
        self.name = name
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.middle_layers = middle_layers
        self.synapses = synapses
        self.time = 0

    def __str__(self):
        header = 'NETWORK \'%s\'\n' % self.name
        return header + ('=' * (len(header)-1)) + '\nLayers:\n' + '\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.middle_layers]) + '\nSynapses:\n' + '\n'.join([str(s) for s in self.synapses])

    def updateInputLayer(self, values):
        if len(self.input_layer.neurons) != len(values):
            raise ValueError('Number of input neurons mismatch in input file.')
        for val, neuron in zip(values, self.input_layer.neurons):
            neuron.f(val)

    def step(self):
        for syn in self.synapses:
            syn.target.f(syn.origin.value)
        print('Status at time %d:' % self.time)
        print('\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.middle_layers]))
        self.time += 1

    def run(self, input_file, time_shift):
        values = [np.asarray(line.split(' ')).astype(float) for line in input_file.read().splitlines()]
        self.updateInputLayer(values[0])
        for i in range(1, time_shift):
            self.step()
            self.updateInputLayer(values[i])
            print('Status at time %d\':' % self.time)
            print('\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.middle_layers]))



    __repr__ = __str__