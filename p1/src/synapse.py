#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Synapse:
    origin = None
    target = None
    weight = None

    def __init__(self, origin, target, weight):
        self.origin = origin
        self.target = target
        self.weight = weight

    def __str__(self):
        return '{} --[{}]-> {}'.format(self.origin, self.weight, self.target)

    __repr__ = __str__
