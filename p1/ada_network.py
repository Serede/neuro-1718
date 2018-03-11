import numpy as np
from network import Network
from dataset import btp, ptb

__default_max_epoch__ = 100
__default_threshold__ = 0.01


class AdaNetwork(Network):
    input_size = None
    output_size = None
    learn_rate = None

    def __init__(self, name, input_size, output_size, learn_rate):
        super(AdaNetwork, self).__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        self.theta = 0

        self.learn_rate = learn_rate
        self.bias = np.random.uniform(-1., 1., output_size)
        self.synapses = np.random.uniform(-1., 1., input_size * output_size).reshape((output_size, input_size))

    def __str__(self):
        return 'Ada-' + super(AdaNetwork, self).__str__()

    def train(self, input_train, output_train, max_epoch=__default_max_epoch__,
              threshold=__default_threshold__):

        output_train_polar = btp(output_train)

        epochs = 0
        delta = np.inf

        while threshold < delta and epochs < max_epoch:
            delta = self.train_all_instances(input_train, output_train_polar)
            epochs += 1

        return

    def train_all_instances(self, input_polar, output_polar):

        delta = -np.inf

        for input_i, output_i in zip(input_polar, output_polar):
            delta = max(delta, self.train_i(input_i, output_i))
        print('SCORE: ', self.score(input_polar, output_polar))

        return delta

    def train_i(self, input_i, output_i):

        y_in = np.dot(self.synapses, input_i) + self.bias

        delta_w = self.learn_rate * np.dot((output_i - y_in).reshape((self.output_size, 1)),
                                           input_i.reshape((1, self.input_size)))
        # print("*"*20)
        # print("t:",output_i)
        # print("y:",y)
        # print("t-y:",(output_i - y))
        # print("t-y:",(output_i - y).reshape((self.output_size, 1)))
        # print("x:",input_i)
        # print("x:",input_i.reshape((1, -1)))
        # print("(t-y)x:",np.dot((output_i - y).reshape((self.output_size, 1)), input_i.reshape((1, -1))))
        # print("a(t-y)x:",delta_w)
        # print(">"*20)

        delta_b = self.learn_rate * (output_i - y_in)

        self.synapses = self.synapses + delta_w
        self.bias = self.bias + delta_b

        max_delta_w = np.max(np.abs(delta_w))
        max_delta_b = np.max(np.abs(delta_b))
        max_delta = max([max_delta_b, max_delta_w])

        return max_delta

    def classify_i_p(self, datain):
        """
        The output is polar
        :param datain:
        :return:
        """
        y_in = np.dot(self.synapses, datain) + self.bias
        y = np.zeros(shape=y_in.shape, dtype=int)

        y[y_in < 0] = -1
        y[y_in > 0] = 1

        return y

    def classify(self, datain, binary_output=True):
        polar = np.vstack(tuple(map(lambda x: self.classify_i_p(x), datain)))
        if binary_output:
            return ptb(polar)
        else:
            return polar

    def score(self, datain, dataout):
        dataout_binary = ptb(dataout)

        res = self.classify(datain, binary_output=True)
        score = (dataout_binary == res).sum() / dataout_binary.size

        return score

    def run(self, datain, binary_output=True, verbose=False):
        return self.classify(datain, binary_output)
