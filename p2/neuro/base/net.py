# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base neural network module.

Todo:
    * Tackle binary/polar schizophrenia.
    * Add method for confusion matrix.
    * Implement __str__.
    * Test everything.
"""

from abc import ABC, abstractmethod

# Use special key None for bias)
_bias_ = None


class Net(ABC):
    """Base neural network class.

    Attributes:
        name (str): Net instance name.
    """

    name = None

    _x = None
    _y = None
    _synapses = None

    def __init__(self, name):
        self.name = name
        self._x = list()
        self._y = list()
        self._synapses = dict()

    def __str__(self):
        pass

    __repr__ = __str__

    def _dfs(self, c, instance, hist=dict()):
        """Internal implementation of Depth First Search for cell transfers.

        Args:
            c (str): Final cell name.
            instance (list): Ordered list of values for input layer.
            hist (dict, optional): Defaults to {}. Value history.

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
            # Add values to history
            hist[c] = [x[c], x[c]]
            # Return input value from instance
            return x[c]
        # Get bias
        b = self._synapses[c][None]
        # Weighted sum of ingoing synapses
        s = sum([w * self._dfs(d, instance, hist=hist)
                 for d, w in self._synapses[c].items()])
        # Add values to history
        hist[c] = [b + s, self.f(b + s)]
        # Return output value
        return self.f(b + s)

    def add_cell(self, name, type=None):
        """Add a new cell to the net.

        Args:
            name (str): Unique name for the new cell.
            type (str, optional): Defaults to None. Cell type ('in'/'out'/None).

        Raises:
            ValueError: If `name` or `type` are invalid.
        """

        # Check if cell already exists
        if name in self._synapses:
            raise ValueError(
                'Cell "{}" already exists in net "{}".'.format(name, self.name))
        # Initialize cell synapses dict
        self._synapses[name] = dict()
        # Initialize bias to 0
        self._synapses[name][_bias_] = 0
        # Mark as input or output cell, if any
        if type == 'in':
            self._x.append(name)
        elif type == 'out':
            self._y.append(name)
        elif type:
            raise ValueError('Invalid cell type "{}".'.format(type))

    def add_cells(self, name, n, type=None):
        """Bulk add new cells to the net.

        Args:
            name (str): Base name for the new cells (sequential numbers will follow).
            n (int): Amount of cells to bulk add.
            type (str, optional): Defaults to None. Cell type ('in'/'out'/None).

        Raises:
            ValueError: If `name` or `type` are invalid.
        """
        for i in range(n):
            # If more than one use base name
            if n > 1:
                self.add_cell('{}{}'.format(name, i), type=type)
            # Else use whole name
            else:
                self.add_cell(name, type=type)

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
            pre (str): Pre-synaptic cell name (base name if `n` > 1).
            post (str): Post-synaptic cell name (base name if `n` > 1).
            weight (float): Synaptic weight.
            n (int, optional): Defaults to 1. Amount of cells in the `pre` set.
            m (int, optional): Defaults to 1. Amount of cells in the `post` set.

        Raises:
            LookupError: If `pre` or `post` are invalid.
        """

        for i in range(n):
            for j in range(m):
                # If more than one use base name
                if n > 1:
                    _pre = '{}{}'.format(pre, i)
                # Else use whole name
                else:
                    _pre = pre
                # If more than one use base name
                if m > 1:
                    _post = '{}{}'.format(post, j)
                # Else use whole name
                else:
                    _post = post
                self.add_synapse(_pre, _post, weight)

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

    def score(self, datain, dataout):
        """Compute ratio of successfully tested data.

        Args:
            datain (list): Ordered list of input instances to test.
            dataout (list): Ordered list of expected output instances.

        Raises:
            ValueError: If `datain` or `dataout` are invalid.

        Returns:
            float: Ratio of sucessfully tested data.
        """

        # Check that data sizes match
        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')
        # Run test for input data
        results = self.test(datain)
        # Get number of instances
        n = len(results)
        # Return success ratio
        return sum([1 for i in range(n) if results[i] == dataout[i]]) / n

    @abstractmethod
    def train(self, datain, dataout, learn, epochs):
        """Adjust net weights from training data.

        Args:
            datain (list): Ordered list of input instances to train.
            dataout (list): Ordered list of expected output instances.
            learn (float): Learning rate to use during training.
            epochs (int): Maximum number of epochs to train.

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
