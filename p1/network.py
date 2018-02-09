#!/usr/bin/env python3

class Network():
    layers = None
    connections = None

    def __init__(self, layers, connections):
        self.layers = layers
        self.connections = connections