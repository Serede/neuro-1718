0#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from copy import deepcopy
from abc import ABC, abstractmethod


class Network(ABC):
    name = None

    def __init__(self, name):
        self.name = name

    def __str__(self):
        header = 'NETWORK \'%s\'\n' % self.name
        return header + ('=' * (len(header)-1))

    __repr__ = __str__

    @abstractmethod
    def run(self, datain, verbose=False):
        pass
