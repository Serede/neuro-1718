#!/usr/bin/env python3
# -*- coding: utf-8 -*-

_pairno_ = '08'
_authors_ = 'Sergio Fuentes, Adrián Muñoz'

import sys
import numpy as np
import argparse as ap
from per_network import PerNetwork


def main():
    parser = ap.ArgumentParser(
        description='Práctica 1 - Perceptrón\nPareja {}: {}'.format(_pairno_, _authors_))
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-v', '--verbose', help='Display network steps',
                        action='store_true', required=False)
    args = vars(parser.parse_args())

    with open(args['input']) as filein:
        datain = [np.asarray(line.split(' ')).astype(float)
                  for line in filein.read().splitlines()]

    network = PerNetwork('PerNetwork', int(
        datain[0][0]), int(datain[0][1]), 10, 0.25)
    print(network)
    network.train(datain[1:], verbose=args['verbose'])

    with open(args['output'], 'w') as fileout:
        dataout = network.run(datain[1:], verbose=args['verbose'])
        for line in dataout:
            fileout.write('{}\n'.format(line))
        print(dataout)


if __name__ == "__main__":
    main()
