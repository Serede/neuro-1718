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

    train_count = None
    test_count = None

    def __init__(self, filename, binary=True, normalize=True):
        with open(filename, 'r') as file:
            # Load metadata
            [len_in, len_out] = file.readline().split()
            self.input_length = int(len_in)
            self.output_length = int(len_out)

            # Load dataset
            data = np.asarray([l.split() for l in file.read().splitlines()]).astype(float)

            if normalize:

                min_ = np.min(data, axis=0)
                data = (data - min_) / (np.max(data, axis=0) - min_)

                self.input_data = data[:, :self.input_length]
                self.output_data = data[:, self.input_length:].astype(int)

                if binary:
                    self.output_data[self.output_data == -1] = 0
                else:
                    self.output_data[self.output_data == 0] = -1

                self.instance_count = int(self.input_data.shape[0])


    def partition(self, ratio):
        last_index_train = int(np.floor(ratio * self.instance_count))

        if not 0 < last_index_train < self.instance_count:
            raise ValueError("Invalid ratio. Empty train or test set.")

        self.train_count = last_index_train
        self.test_count = self.instance_count - last_index_train

        permutation = np.random.permutation(self.instance_count)
        indices_train = permutation[:last_index_train]
        indices_test = permutation[last_index_train:]

        train_input = self.input_data[indices_train]
        train_output = self.output_data[indices_train]

        test_input = self.input_data[indices_test]
        test_output = self.output_data[indices_test]

        return (train_input, train_output), (test_input, test_output)


def errors(self, result_data):
    for i in range(self.instance_count):
        if not (result_data[i] == self.output_data[i]).all():
            print("Mismatch for instance {}: expected {} but got {}.".format(
                i, self.output_data[i], result_data[i]))


def score(self, result_data):
    return (result_data == self.output_data).sum(axis=0) / self.instance_count


def confussion_matrix(self, result_data):
    if result_data.shape != self.output_data.shape:
        raise ValueError(
            "Shape of the result mismatch. Expected {}, got {}".format(self.output_data.shape, result_data.shape))

    # One matrix per output!
    l_matrices = []
    l_values = []

    # for each output
    for output_index in range(self.output_length):

        # Build the dictionary for each output and reate the matrix
        output_expected = self.output_data.T[output_index]
        output_calculated = result_data.T[output_index]

        values = np.sort(np.unique(output_expected))
        matrix = np.zeros(shape=(values.size, values.size))

        indices_dict = {value: index for index, value in enumerate(values)}

        # fill the matrix
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
