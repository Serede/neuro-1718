#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset management.
"""

from math import floor
from random import shuffle


class Dataset:
    """Dataset class.

    Attributes:
        sizein (int): Input layer size.
        sizeout (int): Output layer size.
        count (int): Number of instances.
    """

    sizein = None
    sizeout = None
    count = None

    _datain = None
    _dataout = None

    def __init__(self, filename, pol=False):
        """Constructor.

        Args:
            filename (str): Dataset file.
            pol (bool, optional): Defaults to False. Whether data is in polar form.
        """

        # Open dataset file
        with open(filename, 'r') as file:
            # Load data size
            [_sizein, _sizeout] = file.readline().split()
            self.sizein = int(_sizein)
            self.sizeout = int(_sizeout)
            # Load instances
            if pol:
                data = [[float(x) for x in l.split()]
                        for l in file.read().splitlines()]
            else:
                data = [[2*float(x)-1 for x in l.split()]
                        for l in file.read().splitlines()]
            # Save instance count
            self.count = len(data)
            # Split data
            self._datain = [d[:self.sizein] for d in data]
            self._dataout = [d[self.sizein:] for d in data]

    def data(self):
        """Return a tuple with input and output data.

        Returns:
            tuple: (input data, output data)
        """

        return self._datain, self._dataout

    def partition(self, ratio):
        """Partition data randomly for given ratio.

        Args:
            ratio (float): Ratio for train.

        Raises:
            ValueError: If `ratio` is invalid.

        Returns:
            tuple: (train data, test data)
        """

        # Get index for split
        split = int(floor(ratio * self.count))
        # Check that split index is valid
        if not 0 < split < self.count:
            raise ValueError("Invalid ratio. Empty train or test set.")
        # Create list of indices
        indices = list(range(self.count))
        # Shuffle list of indices
        shuffle(indices)
        # Create train partition
        train = list()
        train.append(list(map(self._datain.__getitem__, indices[:split])))
        train.append(list(map(self._dataout.__getitem__, indices[:split])))
        # Create test partition
        test = list()
        test.append(list(map(self._datain.__getitem__, indices[split:])))
        test.append(list(map(self._dataout.__getitem__, indices[split:])))
        # Return partitions
        return train, test
