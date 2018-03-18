# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monolayer Perceptron implementation.
"""

from neuro.base.net import Net
from doc_inherit import method_doc_inherit


class Perceptron(Net):
    """Monolayer Perceptron class.

    Attributes:
        name (str): Perceptron instance name.
        sizein (int): Input layer size.
        sizeout (int): Output layer size.
        theta (float): Activation threshold.
    """

    name = None
    theta = None

    def __init__(self, name, sizein, sizeout, theta):
        super().__init__(name)
        self.sizein = sizein
        self.sizeout = sizeout
        self.theta = theta
        self.add_cells('x', sizein, type='in')
        self.add_cells('y', sizeout, type='out')
        self.add_synapses('x', 'y', 0, n=sizein, m=sizeout)

    @method_doc_inherit
    def train(self, datain, dataout, learn, epochs):
        # Check that data sizes match
        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')
        # Check learning rate range
        if not 0 < learn <= 1:
            raise ValueError(
                'Learning rate must be within the interval (0, 1].')
        # Initialize stop condition to false
        stop = False
        # Initialize current epoch to zero
        epoch = 0
        # Run epochs until stop conditions are met
        while not stop and epoch < epochs:
            # Assume stop condition
            stop = True
            # For each training pair in data
            for s, t in zip(datain, dataout):
                # Run test for input data
                y = self.test_instance(s)
                # Check that input sizes match
                if len(y) != len(t):
                    raise ValueError(
                        'Instance {} does not match output layer size ({}).'.format(t, len(y)))
                # For each output value obtained
                for j in range(len(y)):
                    # If different from expected value
                    if y[j] != t[j]:
                        _y = self._y[j]
                        # Update bias weight
                        self._synapses[_y][None] += learn * t[j]
                        # For each input value
                        for i in range(len(s)):
                            _x = self._x[i]
                            # Update synaptic weight
                            self._synapses[_y][_x] += learn * t[j] * s[i]
                        # Clear stop condition
                        stop = False
            # Move to next epoch
            epoch += 1

    @method_doc_inherit
    def f(self, y):
        if y < -self.theta:
            return -1
        if y > self.theta:
            return 1
        return 0
