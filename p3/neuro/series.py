# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series implementation.
"""

from copy import deepcopy

from neuro.ml_perceptron import MLPerceptron, BIAS_KEY, NAME_I, NAME_O, NAME_H


class Series(MLPerceptron):
    """Series class.

    Attributes:
        name (str): Perceptron instance name.
        sizein (int): Input layer size.
        sizeout (int): Output layer size.
        hsizes (list): Hidden layers sizes.
    """

    def __init__(self, name, sizein, sizeout, hsizes):
        super().__init__(name, sizein, sizeout, hsizes)

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
        hist[c] = [b + s, b + s]
        # Return output value
        return b + s

    """Adjust net weights from training data (recursively).

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

    def train_recursive(self, datain, dataout, learn, epochs, normalize=False):
        # Check that data sizes match
        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')
        # Check learning rate range
        if not 0 < learn <= 1:
            raise ValueError(
                'Learning rate must be within the interval (0, 1].')
        # Print initial progress
        print('Training... 0%', end='\r')
        # Normalize input if required
        if normalize:
            self.normalize(datain)
        # Create list for MSE
        mse = list()
        # Initialize stop condition to false
        stop = False
        # Initialize current epoch to zero
        epoch = 0
        # Initialize previous value for recursion
        previous = None
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
                if previous is not None:
                    # Recursive case
                    y = self.test_instance(s[:-1] + [previous], hist=hist)
                else:
                    # Basic case
                    y = self.test_instance(s, hist=hist)
                # Save value for next recursion
                previous = y[0]
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
            # Print current progress
            print('Training... {}%'.format(
                int(100 * epoch / epochs)), end='\r', flush=True)
        # Print end of line
        print()
        # Return MSE list
        return mse

    def stats(self, datain, dataout):
        """Gather statistics about test data classification.

        Args:
            datain (list): Ordered list of input instances to test.
            dataout (list): Ordered list of expected output instances.

        Raises:
            ValueError: If `datain` or `dataout` are invalid.

        Returns:
            tuple: (ecm, basic)
        """

        # Check that data sizes match
        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')
        # Run test for input data
        results = self.test(datain)
        # Get number of instances
        n = len(results)
        # Initialize ECM metrics
        ecm = 0
        basic = 0
        # For every instance
        for X, T, Y in zip(datain, dataout, results):
            # Accumulate ECM metrics
            ecm += sum([(t - y) ** 2 for t, y in zip(T, Y)])
            basic += sum([(t - X[-1]) ** 2 for t in T])
        # Return both stats
        return ecm/n, basic/n
