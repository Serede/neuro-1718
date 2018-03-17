# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base neural network module.

Todo:
    * Add bias-related logic.
    * Tackle binary/polar schizophrenia.
    * Add method for confusion matrix.
    * Implement __str__.
    * Test everything.
"""

from abc import ABC, abstractmethod


class Net(ABC):
    """Base neural network class.

    Attributes:
        name (str): Net instance name.
    """

    name = None

    _x = None
    _y = None
    _synapses = None

    def __init__(self, name, layers, f):
        self.name = name
        self._x = list()
        self._y = list()
        self._synapses = dict()

    def __str__(self):
        pass

    __repr__ = __str__

    def add_cell(self, name, type=None):
        """Add a new cell to the net.

        Args:
            name (str): Unique name for the new cell.
            type (str, optional): Defaults to None. Cell type ('in'/'out'/None).

        Raises:
            ValueError: If `name` or `type` are invalid.
        """

        if name in self._synapses:
            raise ValueError(
                'Cell "{}" already exists in net "{}".'.format(name, self.name))
        self._synapses[name] = dict()
        if type == 'in':
            self._x.append(name)
        elif type == 'out':
            self._y.append(name)
        else:
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
            self.add_cell('{}{}'.format(name, i), type=type)

    def add_synapse(self, pre, post, weight):
        """Add a synapse between two cells in the net.

        Args:
            pre (str): Pre-synaptic cell name.
            post (str): Post-synaptic cell name.
            weight (float): Synaptic weight.

        Raises:
            LookupError: If `pre` or `post` are invalid.
        """

        if not pre in self._synapses:
            raise LookupError(
                'Cell "{}" does not exist in the net.'.format(pre))
        if not post in self._synapses:
            raise LookupError(
                'Cell "{}" does not exist in the net.'.format(post))
        self._synapses[pre][post] = weight

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
                if n > 1:
                    _pre = '{}{}'.format(pre, i)
                else:
                    _pre = pre
                if m > 1:
                    _post = '{}{}'.format(post, j)
                else:
                    _post = post
                self.add_synapse(_pre, _post, weight)

    def test_instance(self, instance):
        """Run the net with an instance as input.

        Args:
            instance (list): Ordered list of values for input layer.

        Raises:
            ValueError: If `instance` is invalid.

        Returns:
            list: Ordered list of values from output layer.
        """

        if len(instance) != len(self._x):
            raise ValueError('Instance {} does not match input layer size ({}).'.format(
                instance, len(self._x)))
        # Create dict for storing cell values
        v = dict()
        # Set input layer values from instance
        v.update({name: value for name, value in zip(self._x, instance)})
        # Set initial output layer values to zero
        v.update({name: 0 for name in self._y})
        # Initialize synaptic queue with input cells
        q = self._x
        # Process synaptic queue until empty
        while q:
            # Get first cell in queue
            pre = q.pop(0)
            # For each outgoing synapse
            for post, weight in self._synapses[pre].items():
                # Update post-synaptic cell value
                v[post] = self.f(weight * v[pre])
                # Append post-synaptic cell to queue
                q.append(post)
        # Return output layer values
        return [v[y] for y in self._y]

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

        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')
        results = self.test(datain)
        n = len(results)
        return sum([1 for i in range(n) if results[i] == dataout[i]]) / n

    @abstractmethod
    def train(self, datain, dataout, learn):
        """Adjust net weights from training data.

        Args:
            datain (list): Ordered list of input instances to train.
            dataout (list): Ordered list of expected output instances.
            learn (float): Learning rate to use during training.

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
            int: Output.
        """

        pass
