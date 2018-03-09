#!/usr/bin/env python3
# -*- coding: utf-8 -*-

_pairno_ = '08'
_authors_ = 'Sergio Fuentes, Adrián Muñoz'

import sys
import numpy as np
import argparse as ap


def main():
    parser = ap.ArgumentParser(
        description='Práctica 1 - Adaline\nPareja {}: {}'.format(_pairno_, _authors_))
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-v', '--verbose', help='Display network steps',
                        action='store_true', required=False)
    args = vars(parser.parse_args())


    with open(args['input']) as filein, open(args['output'], 'w') as fileout:

        datain = [np.asarray(line.split(' ')).astype(float)
                  for line in filein.read().splitlines()]
        print(datain)


if __name__ == "__main__":
    main()
