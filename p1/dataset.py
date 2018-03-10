#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Dataset:

    input_length = None
    output_length = None
    input_data = None
    output_data = None
    instance_count = None

    def __init__(self, filename, binary=True):
        with open(filename, 'r') as file:
            # Load metadata
            [len_in, len_out] = file.readline().split()
            self.input_length = int(len_in)
            self.output_length = int(len_out)

            # Load dataset
            data = np.asarray([l.split() for l in file.read().splitlines()])

            self.input_data = data[:, :self.input_length].astype(float)
            self.output_data = data[:, self.input_length:].astype(int)
            if binary:
                self.output_data = 2*self.output_data - 1

            self.instance_count = int(self.input_data.shape[0])

    def partition(self, ratio):
        pass

    def errors(self, result_data):
        for i in range(self.instance_count):
            if not (result_data[i] == self.output_data[i]).all():
                print("Mismatch for instance {}: expected {} but got {}.".format(
                    i, self.output_data[i], result_data[i]))

    def score(self, result_data):
        return (result_data == self.output_data).sum(axis=0) / self.instance_count
