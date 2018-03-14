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
        head = self.__class__.__name__ + " '{}':\n".format(self.name)
        body = '\n'.join(['  + ' + str(n) for n in self.neurons])
        return head + body

    __repr__ = __str__
