# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multilayer Perceptron implementation.
"""

from math import exp
from statistics import mean, stdev
from copy import deepcopy
try:
    from doc_inherit import method_doc_inherit
except ImportError:
    # Dummy decorator (docstrings will not be inherited)
    def method_doc_inherit(func):
        return func

from neuro.base.net import Net, BIAS_KEY

# Input cells basename
NAME_I = 'x'
# Output cells basename
NAME_O = 'y'
# Hidden cells basename
NAME_H = 'z'


class MLPerceptron(Net):
    """Multilayer Perceptron class.

    Attributes:
        name (str): Perceptron instance name.
        sizein (int): Input layer size.
        sizeout (int): Output layer size.
        hsizes (list): Hidden layers sizes.
    """

    name = None
    sizein = None
    sizeout = None
    hsizes = None
    _hnames = None

    def __init__(self, name, sizein, sizeout, hsizes):
        if not hsizes:
            raise ValueError('Invalid hidden layers sizes.')
        super().__init__(name)
        self.sizein = sizein
        self.sizeout = sizeout
        self.hsizes = hsizes
        self._hnames = []
        # For each hidden layer
        for i in range(len(hsizes)):
            # Create base name
            self._hnames.append(NAME_H * (i + 1))
            # Add hidden cells
            self.add_cells(self._hnames[i], hsizes[i])
            # Add synapses with previous hidden layer
            if i > 0:
                self.add_synapses(self._hnames[i-1], self._hnames[i],
                                  0, n=hsizes[i-1], m=hsizes[i])
        # Add input layer cells
        self.add_cells(NAME_I, sizein, type='in')
        # Add output layer cells
        self.add_cells(NAME_O, sizeout, type='out')
        # Add input synapses
        self.add_synapses(NAME_I, self._hnames[0], 0, n=sizein, m=hsizes[0])
        # Add output synapses
        self.add_synapses(self._hnames[i], NAME_O, 0, n=hsizes[i], m=sizeout)

    @method_doc_inherit
    def train(self, datain, dataout, learn, epochs, normalize=False):
        # Check that data sizes match
        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')
        # Check learning rate range
        if not 0 < learn <= 1:
            raise ValueError(
                'Learning rate must be within the interval (0, 1].')
        # Normalize input if required
        if normalize:
            self.normalize(datain)
        # Create list for MSE
        mse = list()
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
                synapses = deepcopy(self._synapses)
                # Create history dict
                hist = dict()
                # Populate history dict and get output values
                y = self.test_instance(s, hist=hist)
                # Initialize list of input deltas
                δ_in = [t[j] - y[j] for j in range(len(y))]
                # Create lists for retropropagation
                names = [NAME_O] + self._hnames[::-1] + [NAME_I]
                sizes = [self.sizeout] + self.hsizes[::-1] + [self.sizein]
                # Retropropagation
                for k in range(len(names) - 1):
                    # Initialize list for previous layer δ_in
                    _δ_in = [0] * sizes[k + 1]
                    # For each cell in current layer
                    for j in range(sizes[k]):
                        # If correction is needed
                        if δ_in[j] != 0:
                            _zz = '{}{}'.format(names[k], j)
                            # Get input value from history
                            zz_in = hist[_zz][0]
                            # Compute current delta
                            δ = δ_in[j] * self.df(zz_in)
                            # Save bias weight correction
                            synapses[_zz][BIAS_KEY] += learn * δ
                            # For each cell in previous layer
                            for i in range(sizes[k + 1]):
                                _z = '{}{}'.format(names[k + 1], i)
                                # Get output value from history
                                z = hist[_z][1]
                                # Save synaptic weight correction
                                synapses[_zz][_z] += learn * δ * z
                                # Accumulate δ_in
                                _δ_in[i] += δ * self._synapses[_zz][_z]
                            # Clear stop condition
                            stop = False
                    # Set input deltas for previous layer
                    δ_in = _δ_in
                # Update synapses
                self._synapses = synapses
            # Test dataset at the end of current epoch
            Y = self.test(datain)
            # Compute residual sum of squares
            rss = sum([sum([(dataout[i][j] - Y[i][j]) **
                            2 for j in range(len(Y[i]))]) / len(Y[0]) for i in range(len(Y))])
            # Append MSE value after current epoch
            mse.append(rss / len(Y))
            # Move to next epoch
            epoch += 1
        # Return MSE list
        return mse

    @method_doc_inherit
    def f(self, y):
        return 2 / (1 + exp(-y)) - 1

    def df(self, y):
        """Net-wide transfer function derivative.

        Args:
            y (float): Input signal.

        Returns:
            float: Output.
        """

        return (1 + self.f(y)) * (1 - self.f(y)) / 2
