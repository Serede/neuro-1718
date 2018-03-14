#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, RawTextHelpFormatter

from __metadata__ import _COURSE_, _YEAR_, _TITLE_, _PAIR_, _AUTHORS_
from src.mode_parser import build_mode_subparser, mode1, mode2, mode3
from src.ada_network import AdaNetwork
from src.dataset import Dataset

_input_ = 0
_output_ = 1


def build_parser():
    description = '{} {}\n{}. Adaline\nPareja {}: {}'.format(
        _COURSE_, _YEAR_, _TITLE_, _PAIR_, ' y '.join(_AUTHORS_))

    parser = ArgumentParser(description=description,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-t', '--threshold', help='Tolerance for weight updates', required=True)
    parser.add_argument(
        '-l', '--learn', help='Learning rate', required=True)
    parser.add_argument(
        '-e', '--epochs', help='Maximum number of epochs to train', required=True)
    parser.add_argument(
        '-v', '--verbose', help='Display steps', action='store_true')

    return parser


def main():
    size = None
    train_data = None
    test_data = None

    parser = build_parser()
    build_mode_subparser(parser)

    args = vars(parser.parse_args())

    epoch = float(args['epoch'])
    threshold = float(args['threshold'])
    learn = float(args['learn'])

    if args['mode'] == 'mode1':
        size, train_data, test_data = mode1(args['data'], args['ratio'])

    elif args['mode'] == 'mode2':
        size, train_data, test_data = mode2(args['data'])

    elif args['mode'] == 'mode3':
        size, train_data, test_data = mode3(args['train'], args['test'])

    network = AdaNetwork(
        'AdaNetwork', size[_input_], size[_output_], learn_rate=learn)
    print(network)

    network.train(train_data[_input_], train_data[_output_],
                  max_epoch=epoch, threshold=threshold)
    network.run(test_data[_input_])


if __name__ == "__main__":
    main()