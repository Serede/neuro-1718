# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multilayer Perceptron implementation.
"""

import math
from doc_inherit import method_doc_inherit

from neuro.base.net import Net


class MLPerceptron(Net):
    """Multilayer Perceptron class.

    Attributes:
        name (str): Perceptron instance name.
        sizein (int): Input layer size.
        sizeout (int): Output layer size.
        hsizes (tuple): Hidden layers sizes.
        theta (float): Activation threshold.
    """

    name = None
    sizein = None
    sizeout = None
    hsizes = None
    theta = None
    _hnames = None

    def __init__(self, name, sizein, sizeout, hsizes, theta):
        if not hsizes:
            raise ValueError('Invalid hidden layers hsizes.')
        super().__init__(name)
        self.sizein = sizein
        self.sizeout = sizeout
        self.hsizes = hsizes
        self.theta = theta
        self._hnames = []
        # For each hidden layer
        for i in range(len(hsizes)):
            self._hnames.append('z' * (i + 1))
            # Add hidden cells
            self.add_cells(self._hnames[i], hsizes[i])
            # Add synapses with previous hidden layer
            if i > 0:
                self.add_synapses(self._hnames[i-1], self._hnames[i],
                                  0, n=hsizes[i-1], m=hsizes[i])
        # Add input layer cells
        self.add_cells('x', sizein, type='in')
        # Add output layer cells
        self.add_cells('y', sizeout, type='out')
        # Add input synapses
        self.add_synapses('x', self._hnames[0], 0, n=sizein, m=hsizes[0])
        # Add output synapses
        self.add_synapses(self._hnames[i], 'y', 0, n=hsizes[i], m=sizeout)

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
                # Check that output sizes match
                if len(self._y) != len(t):
                    raise ValueError(
                        'Instance {} does not match output layer size ({}).'.format(t, len(self._y)))
                # Create dict for new synapses
                new_synapses = {a: {b: w for b, w in d}
                                for a, d in self._synapses}
                # Run test for input data
                y = self.test_instance(s)
                # For each output value obtained
                for j in range(len(y)):
                    # If different from expected value
                    if y[j] != t[j]:
                        _y = self._y[j]
                        # Obtain input value for y
                        y_in = self._dfs(_y, s)
                        # Compute delta
                        δ = (t[j] - y[j]) * self.df(y_in)
                        # Update bias weight
                        new_synapses[_y][None] += learn * δ
                        # For each input value
                        for i in range(len(s)):
                            _x = self._x[i]
                            # Update synaptic weight
                            new_synapses[_y][_x] += learn * t[j] * s[i]
                        # Clear stop condition
                        stop = False
            # Move to next epoch
            epoch += 1

    @method_doc_inherit
    def f(self, y):
        return 2 / (1 + math.exp(-y)) - 1

    def df(self, y):
        """Net-wide transfer function derivative.

        Args:
            y (float): Input signal.

        Returns:
            float: Output.
        """

        return (1 + self.f(y)) * (1 - self.f(y)) / 2
