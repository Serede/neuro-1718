#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class Neuron(ABC):
    name = None

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.__class__.__name__ + str(self.__dict__).replace("'", "")

    __repr__ = __str__

    @abstractmethod
    def f(self, y_in):
        pass
