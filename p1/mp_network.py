#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from copy import deepcopy
from network import Network


class MPNetwork(Network):
    input_layer = None
    output_layer = None
    hidden_layers = None
    synapses = None
    time = None

    def __init__(self, name, input_layer, output_layer, hidden_layers, synapses):
        super(MPNetwork, self).__init__(name)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.synapses = defaultdict(list)
        for syn in synapses:
            self.synapses[syn.target].append(syn)
        self.time = 0

    def __str__(self):
        head = super(MPNetwork, self).__str__()
        layers_head = 'Layers:'
        input_layer = '- ' + str(self.input_layer)
        output_layer = '- ' + str(self.output_layer)
        hidden_layers = '\n'.join(['- ' + str(h) for h in self.hidden_layers])
        layers_body = '\n'.join([input_layer, output_layer, hidden_layers])
        layers = '\n'.join([layers_head, layers_body])
        synapses_head = 'Synapses:'
        synapses_body = '\n'.join(['- ' + str(s)
                                   for s in sum(self.synapses.values(), [])])
        synapses = '\n'.join([synapses_head, synapses_body])
        body = '\n'.join([layers, synapses])
        return '\n'.join([head, body])

    def feel(self, values):
        if len(self.input_layer) != len(values):
            raise ValueError(
                'Number of input neurons mismatch in values {}.'.format(values))
        for val, neuron in zip(values, self.input_layer.neurons):
            neuron.f(val)

    def think(self):
        syn = {k: deepcopy(v) for k, v in self.synapses.items()}
        for target in self.synapses.keys():
            target.f(sum([s.weight * s.origin.value for s in syn[target]]))

    def run(self, datain, verbose=False):
        dataout = []
        self.time = 0
        self.feel(datain[0])
        for i in range(1, len(datain) + len(self.hidden_layers)):
            self.think()
            if(i < len(datain)):
                self.feel(datain[i])
            else:
                self.feel([0] * len(self.input_layer))
            self.time += 1
            if verbose:
                print('Status at time %d:' % self.time)
                input_layer = '- ' + str(self.input_layer)
                output_layer = '- ' + str(self.output_layer)
                hidden_layers = '\n'.join(['- ' + str(h)
                                           for h in self.hidden_layers])
                print('\n'.join([input_layer, output_layer, hidden_layers]))
            dataout.append(' '.join([str(n.value)
                                     for n in self.output_layer.neurons]))
        return dataout
