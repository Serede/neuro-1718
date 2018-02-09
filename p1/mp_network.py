#!/usr/bin/env python3

import numpy as np
from network import Network

class MPNetwork(Network):
    def __init__(n, m, w, p):
        self.n = n
        self.m = m
        self.w = w
        self.p = p