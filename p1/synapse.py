#!/usr/bin/env python3

class Synapse:
    origin = None
    target = None
    weight = None

    def __init__(self, origin, target, weight):
        self.origin = origin
        self.target = target
        self.weight = weight