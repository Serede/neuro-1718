import numpy as np
from network import Network

__default_max_epoch__ = 100
__tolerance_threshold__ = 0.05


class AdaNetwork(Network):
    input_size = None
    output_size = None
    learn_rate = None

    def __init__(self, name, input_size, output_size, learn_rate):
        super(AdaNetwork, self).__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        self.theta = 0

        self.input = np.zeros(shape=(input_size,))
        self.output = np.zeros(shape=(output_size,))

        self.learn_rate = learn_rate
        self.bias = np.zeros(shape=(output_size,))
        self.weights = np.zeros(shape=(output_size, input_size))

    def __str__(self):
        return 'Ada-' + super(AdaNetwork, self).__str__()

    def train(self, datain_train, expected_result_train, max_epoch=__default_max_epoch__,
              tolerance_threshold=__tolerance_threshold__):

        epochs = 0
        error_percentage = 1.

        while tolerance_threshold < error_percentage and epochs < max_epoch:
            self.train_all_instances(datain_train, expected_result_train)
            epochs += 1

        return

    def train_all_instances(self, instances_attributes, expected_results):
        for instance_attributtes, instance_result in zip(instances_attributes, expected_results):
            self.train_one_instance(instance_attributtes, instance_result)

    def train_one_instance(self, attributtes, expected_result):
        self.set_input(attributtes)
        self.calculate()
        self.adjust(expected_result)

    def set_input(self,attributes):
        self.input = attributes

    def calculate(self):
        self.output = (0<= (np.dot(self.weights,self.input) + self.bias))
        return self.output

    def adjust(self,expected_result):
        # wi(nuevo) = wi(anterior)+ a(t-y_in)x

        self.weights = self.weights + self.learn_rate*np.dot((expected_result - self.output), self.input)
