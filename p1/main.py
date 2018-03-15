# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from neuro.mp_net import MPNet


def main():
    net = MPNet('name', (3, 3, 6, 2), 2)
    net.weights = np.matrix('0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 2 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 2 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 2 0 0 0 0 0 0 0 0 0 0 0; 1 0 0 0 1 0 0 0 0 0 0 0 0 0; 0 1 0 0 0 1 0 0 0 0 0 0 0 0; 0 0 1 1 0 0 0 0 0 0 0 0 0 0; 1 0 0 0 0 1 0 0 0 0 0 0 0 0; 0 1 0 1 0 0 0 0 0 0 0 0 0 0; 0 0 1 0 1 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 2 2 2 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 2 2 2 0 0')
    print('dict', net.__dict__, '\n')
    data = np.matrix('1 0 0; 0 1 0; 0 0 1; 1 0 0')
    print('data', data, '\n')
    print(net.test(data))


if __name__ == "__main__":
    main()
