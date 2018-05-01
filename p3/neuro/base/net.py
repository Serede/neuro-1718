# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base neural network module.
"""

from abc import ABC, abstractmethod
from random import uniform
from statistics import mean, stdev

# Use special key None for bias weights
BIAS_KEY = None


class Net(ABC):
    """Base neural network class.

    Attributes:
        name (str): Net instance name.
    """

    name = None

    _x = None
    _y = None
    _synapses = None
    _normalize = None
    _μ = None
    _σ = None

    def __init__(self, name):
        self.name = name
        self._x = list()
        self._y = list()
        self._synapses = dict()
        self._normalize = False
        self._μ = list()
        self._σ = list()

    def _dfs(self, c, instance, hist=dict()):
        """Internal implementation of Depth First Search for cell transfers.

        Args:
            c (str): Final cell name.
            instance (list): Ordered list of values for input layer.
            hist (dict, optional): Defaults to {}. Value history [in, out].

        Raises:
            ValueError: If `instance` is invalid.

        Returns:
            int: Final cell output value.
        """

        # Check that input sizes match
        if len(instance) != len(self._x):
            raise ValueError('Instance {} does not match input layer size ({}).'.format(
                instance, len(self._x)))
        # Create dict with input values from instance
        x = {name: value for name, value in zip(self._x, instance)}
        # Degenerate case
        if not c:
            return 0
        # If input cell
        if c in self._x:
            # Get input value from instance
            _x = x[c]
            # If normalization enabled
            if self._normalize:
                # Get input cell index
                i = self._x.index(c)
                # Normalize value
                _x = (_x - self._μ[i]) / self._σ[i]
            # Add value to history (duplicated for data consistency)
            hist[c] = [_x, _x]
            # Return value
            return _x
        # Get bias
        b = self._synapses[c][BIAS_KEY]
        # Weighted sum of ingoing synapses
        s = sum([w * self._dfs(d, instance, hist=hist)
                 for d, w in self._synapses[c].items() if d is not BIAS_KEY])
        # Add values to history
        hist[c] = [b + s, self.f(b + s)]
        # Return output value
        return self.f(b + s)

    def add_cell(self, basename, i, type=None):
        """Add a new cell to the net.

        Args:
            basename (str): Base name for the new cell.
            i (int): Number to append to the base name.
            type (str, optional): Defaults to None. Cell type ('in'/'out'/None).

        Raises:
            ValueError: If `basename` or `type` are invalid.
        """
        # Create full cell name
        name = '{}{}'.format(basename, i)
        # Check if cell already exists
        if name in self._synapses:
            raise ValueError(
                'Cell "{}" already exists in net "{}".'.format(name, self.name))
        # Initialize cell synapses dict
        self._synapses[name] = dict()
        # Initialize bias to 0
        self._synapses[name][BIAS_KEY] = 0
        # Mark as input or output cell, if any
        if type == 'in':
            self._x.append(name)
            self._μ.append(0)
            self._σ.append(1)
        elif type == 'out':
            self._y.append(name)
        elif type:
            raise ValueError('Invalid cell type "{}".'.format(type))

    def add_cells(self, basename, n, type=None):
        """Bulk add new cells to the net.

        Args:
            basename (str): Base name for the new cells (sequential numbers will follow).
            n (int): Amount of cells to bulk add.
            type (str, optional): Defaults to None. Cell type ('in'/'out'/None).

        Raises:
            ValueError: If `basename` or `type` are invalid.
        """
        for i in range(n):
            self.add_cell(basename, i, type=type)

    def add_synapse(self, pre, post, weight):
        """Add a synapse between two cells in the net.

        Args:
            pre (str): Pre-synaptic cell name.
            post (str): Post-synaptic cell name.
            weight (float): Synaptic weight.

        Raises:
            LookupError: If `pre` or `post` are invalid.
        """

        # Check that cells exist
        if not pre in self._synapses:
            raise LookupError(
                'Cell "{}" does not exist in the net.'.format(pre))
        if not post in self._synapses:
            raise LookupError(
                'Cell "{}" does not exist in the net.'.format(post))
        # Add synapse to dict
        self._synapses[post][pre] = weight

    def add_synapses(self, pre, post, weight, n=1, m=1):
        """Bulk adds synapses between two sets of cells in the net.

        Args:
            pre (str): Pre-synaptic cell base name.
            post (str): Post-synaptic cell base name.
            weight (float): Synaptic weight.
            n (int, optional): Defaults to 1. Amount of cells in the `pre` set.
            m (int, optional): Defaults to 1. Amount of cells in the `post` set.

        Raises:
            LookupError: If `pre` or `post` are invalid.
        """

        for i in range(n):
            for j in range(m):
                _pre = '{}{}'.format(pre, i)
                _post = '{}{}'.format(post, j)
                self.add_synapse(_pre, _post, weight)

    def randomize_synapses(self, low, high):
        """Randomize synaptic weights between given values.

        Args:
            low (float): Minimum weight value.
            high (float): Maximum weight value.
        """

        # For each synapse in the net
        for b, d in self._synapses.items():
            for a in d.keys():
                # Assume weight 0
                w = 0
                # Generate random non-zero weight
                while w == 0:
                    w = uniform(low, high)
                # Set synaptic weight
                self._synapses[b][a] = w

    def normalize(self, data=None):
        """Enable (or disable) input normalization.

        Args:
            data (list, optional): Defaults to None. List of input instances to normalize, or None to disable normalization.

        Raises:
            ValueError: If `data` is invalid.
        """

        # Disable normalization if data is None
        if data is None:
            self._normalize = False
            return

        # Enable normalization
        self._normalize = True
        # Watch out for a size mismatch
        try:
            # For every input cell
            for i in range(len(self._x)):
                # Save mean value
                self._μ[i] = mean(x[i] for x in data)
                # Save standard deviation
                self._σ[i] = stdev(x[i] for x in data)
                if self._σ[i] == 0:
                    self._σ[i] = 1
        # Catch input size mismatch exceptions
        except IndexError:
            raise ValueError(
                'Input instance size mismatch ({}).'.format(len(self._x)))

    def test_instance(self, instance, hist=dict()):
        """Run the net with an instance as input.

        Args:
            instance (list): Ordered list of values for input layer.
            hist (dict, optional): Defaults to {}. Value history.

        Raises:
            ValueError: If `instance` is invalid.

        Returns:
            list: Ordered list of values from output layer.
        """

        # Return output layer values after DFS
        return [self._dfs(y, instance, hist) for y in self._y]

    def test(self, data):
        """Run the net for an ordered list of instances.

        Args:
            data (list): Ordered list of input instances.

        Raises:
            ValueError: If `data` is invalid.

        Returns:
            list: Ordered list of output instances.
        """

        return [self.test_instance(instance) for instance in data]

    def stats(self, datain, dataout):
        """Gather statistics about test data classification.

        Args:
            datain (list): Ordered list of input instances to test.
            dataout (list): Ordered list of expected output instances.

        Raises:
            ValueError: If `datain` or `dataout` are invalid.

        Returns:
            tuple: (score, confussion_matrix)
        """

        # Check that data sizes match
        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')
        # Run test for input data
        results = self.test(datain)
        # Get number of instances
        n = len(results)
        # Initialize score
        score = 0
        # Create confussion matrix
        r = range(len(self._y))
        m = [[0 for _ in r] for _ in r]
        # For every instance
        for T, Y in zip(dataout, results):
            # Get expected class
            i = T.index(max(T))
            # Get predicted class
            j = Y.index(max(Y))
            # If correct
            if i == j:
                # Add to score
                score += 1
            # Accumulate in confussion matrix
            m[i][j] += 1
        # Return both stats
        return score/n, m

    @abstractmethod
    def train(self, datain, dataout, learn, epochs, normalize=False):
        """Adjust net weights from training data.

        Args:
            datain (list): Ordered list of input instances to train.
            dataout (list): Ordered list of expected output instances.
            learn (float): Learning rate to use during training.
            epochs (int): Maximum number of epochs to train.
            normalize (bool, optional): Defaults to False. Normalize data.

        Returns:
            list: Output MSE value throughout the epochs.

        Raises:
            ValueError: If `datain` or `dataout` are invalid.
        """

        pass

    @abstractmethod
    def f(self, y):
        """Net-wide transfer function.

        Args:
            y (float): Input signal.

        Returns:
            float: Output.
        """

        pass
