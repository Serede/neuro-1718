0  # !/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class Network(ABC):
    name = None

    def __init__(self, name):
        self.name = name

    def __str__(self):
        header = self.__class__.__name__ + " '{}'".format(self.name)
        bar = '=' * len(header)
        return '\n'.join([bar, header, bar])

    __repr__ = __str__

    @abstractmethod
    def run(self, datain, verbose=False):
        pass
