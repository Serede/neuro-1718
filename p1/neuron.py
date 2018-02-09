#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod

class Neuron(metaclass=ABCMeta):
    name = None

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '[%s]' % self.name

    __repr__ = __str__

    @abstractmethod
    def f(self, y_in):
        pass