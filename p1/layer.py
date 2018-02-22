#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Layer:
    name = None
    neurons = None

    def __init__(self, name, neurons):
        self.name = name
        self.neurons = neurons

    def __len__(self):
        return len(self.neurons)

    def __str__(self):
        return '- %s Layer:\n' % self.name + '\n'.join(['  + ' +str(n) for n in self.neurons])

    __repr__ = __str__