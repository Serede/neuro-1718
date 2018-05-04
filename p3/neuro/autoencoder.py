# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autoencoder implementation.
"""

from neuro.ml_perceptron import MLPerceptron

# Use special key None for bias weights
BIAS_KEY = None


class Autoencoder(MLPerceptron):
    """Autoencoder class.

    Attributes:
        name (str): Perceptron instance name.
        sizein (int): Input layer size.
        sizeout (int): Output layer size.
        hsizes (list): Hidden layers sizes.
    """

    def __init__(self, name, sizein, sizeout, hsizes):
        super().__init__(name, sizein, sizeout, hsizes)

    def test(self, data):
        """Run the net for an ordered list of instances.

        Args:
            data (list): Ordered list of input instances.

        Raises:
            ValueError: If `data` is invalid.

        Returns:
            list: Ordered list of output instances.
        """

        return [list(map(lambda x: 2*int(x>0)-1, d)) for d in super().test(data)]

    def stats(self, datain, dataout):
        """Gather statistics about test data classification.

        Args:
            datain (list): Ordered list of input instances to test.
            dataout (list): Ordered list of expected output instances.

        Raises:
            ValueError: If `datain` or `dataout` are invalid.

        Returns:
            tuple: (wrong_outputs, mean_wrong_output, correct_instances)
        """

        # Check that data sizes match
        if len(datain) != len(dataout):
            raise ValueError('Input and output instance counts do not match.')

        # Run test for input data
        results = self.test(datain)
        # Get number of instances
        n = len(results)
        # Initialize metrics
        wo = 0

        ci = 0

        # For every instance
        for T, Y in zip(dataout, results):
            e = sum([t != y for t, y in zip(T, Y)])
            print(10 * '*')
            print('Diff: ', [int(t != y) for t, y in zip(T, Y)])
            print(10 * '*')

            # Check if the output is correct
            if e == 0:
                ci += 1
            # Otherwise, gather errors
            else:
                wo += e

        return wo, wo / len(datain), ci