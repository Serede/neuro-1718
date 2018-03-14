#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np


class Dataset:
    input_length = None
    output_length = None
    instance_count = None

    input_data = None
    output_data = None

    def __init__(self, filename, normalize=True, binary=True):
        with open(filename, 'r') as file:
            # Load metadata
            [len_in, len_out] = file.readline().split()
            self.input_length = int(len_in)
            self.output_length = int(len_out)

            # Load dataset
            data = np.asarray([l.split() for l in file.read().splitlines()])

            self.instance_count = len(data)

            self.input_data = data[:, :self.input_length].astype(float)
            if normalize:
                min_ = np.min(self.input_data, axis=0)
                max_ = np.max(self.input_data, axis=0)
                self.input_data = (self.input_data - min_) / (max_ - min_)

            self.output_data = data[:, self.input_length:].astype(int)
            if binary:
                self.output_data = btp(self.output_data)

    def partition(self, ratio):
        split_index = int(np.floor(ratio * self.instance_count))

        if not 0 < split_index < self.instance_count:
            raise ValueError("Invalid ratio. Empty train or test set.")

        perm = np.random.permutation(self.instance_count)
        i_train = perm[:split_index]
        i_test = perm[split_index:]

        train_in = self.input_data[i_train]
        train_out = self.output_data[i_train]

        test_in = self.input_data[i_test]
        test_out = self.output_data[i_test]

        return (train_in, train_out), (test_in, test_out)

    def errors(self, data):
        for i in range(min(data, self.instance_count)):
            if not (data[i] == self.output_data[i]).all():
                print("Mismatch for instance {}: expected {} but got {}.".format(
                    i, self.output_data[i], data[i]))

    def score(self, data):
        return (data == self.output_data).sum(axis=0) / self.instance_count

    def confussion_matrix(self, data):
        if data.shape != self.output_data.shape:
            raise ValueError(
                "Shape of the result mismatch. Expected {}, got {}".format(self.output_data.shape, data.shape))

        # One matrix per output!
        l_matrices = []
        l_values = []

        # For each output
        for i in range(self.output_length):

            # Build the dictionary for each output and create the matrix
            output_expected = self.output_data.T[i]
            output_calculated = data.T[i]

            values = np.sort(np.unique(output_expected))
            matrix = np.zeros(shape=(values.size, values.size))

            indices_dict = {value: index for index, value in enumerate(values)}

            # Fill the matrix
            for j in range(self.instance_count):
                index_expected = indices_dict[output_expected[j]]
                index_calculated = indices_dict[output_calculated[j]]
                matrix[index_expected][index_calculated] += 1

            l_matrices.append(matrix)
            l_values.append(values)

        return l_matrices, l_values


def btp(vector):
    polar = deepcopy(vector)

    polar[polar == 0] = -1

    return polar


def ptb(vector):
    binary = deepcopy(vector)

    binary[binary == -1] = 0

    return binary
