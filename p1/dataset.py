import numpy as np


class Dataset:

    input_size = None
    output_size = None
    number_of_instances = None

    input = None
    output = None

    def make_partitions(self):
        pass

    def load(self, file_name):
        with open(file_name, 'r') as file:
            # load metadata
            shape = np.asarray(file.readline().split())
            self.input_size = int(shape[0])
            self.output_size = int(shape[1])

            # Load the dataset
            data = np.asarray([line.split() for line in file.read().splitlines()])

            self.input = data[:, :self.input_size].astype(float)
            self.output = data[:, self.input_size:].astype(int)

            self.number_of_instances = int(self.input.shape[0])

    def errors(self,results):

        for index in range(self.number_of_instances):
            if not (results[index] == self.output[index]).all():
                print("Mismatch in instance {}: expected {}, got {}.".format(index, self.output[index], results[index]))

    def score(self,results):
        return (results == self.output).sum(axis=0) / self.number_of_instances
