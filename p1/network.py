#!/usr/bin/env python3

class Network():
    name = None
    layers = None
    synapses = None

    def __init__(self, name, layers, synapses):
        self.name = name
        self.layers = layers
        self.synapses = synapses

    def __str__(self):
        header = 'NETWORK \'%s\'\n' % self.name
        return header + ('=' * (len(header)-1)) + '\nLayers:\n' + '\n'.join([str(l) for l in self.layers]) + '\nSynapses:\n' + '\n'.join([str(s) for s in self.synapses])

    __repr__ = __str__