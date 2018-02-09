#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod

class Neuron(metaclass=ABCMeta):
    @abstractmethod
    def f(self, y_in):
        pass