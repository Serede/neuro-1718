#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from copy import deepcopy
from abc import ABC, abstractmethod


class Network(ABC):
    name = None
    input_layer = None
    output_layer = None
    hidden_layers = None
    synapses = None

    def __init__(self, name, input_layer, output_layer, hidden_layers, synapses):
        self.name = name

    def __str__(self):
        header = 'NETWORK \'%s\'\n' % self.name
        return header + ('=' * (len(header)-1)) + '\nLayers:\n' + '\n'.join([str(l) for l in [self.input_layer, self.output_layer] + self.hidden_layers]) + '\nSynapses:\n' + '\n'.join([str(s) for s in sum(self.synapses.values(), [])])

    __repr__ = __str__

    @abstractmethod
    def run(self, datain, verbose=False):
        pass
