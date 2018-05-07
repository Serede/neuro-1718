# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series implementation.
"""

from neuro.ml_perceptron import MLPerceptron, BIAS_KEY


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
