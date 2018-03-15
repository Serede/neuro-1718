#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, RawTextHelpFormatter

from __metadata__ import _COURSE_, _YEAR_, _TITLE_, _PAIR_, _AUTHORS_
from src.mode_parser import build_mode_subparser, mode1, mode2, mode3
from src.ada_network import AdaNetwork
from src.dataset import Dataset

_in_ = 0
_out_ = 1


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
        '-e', '--epoch', help='Maximum number of epochs to train', required=True)

    return parser


def main():
    parser = build_parser()
    build_mode_subparser(parser)

    args = vars(parser.parse_args())

    epoch = float(args['epoch'])
    threshold = float(args['threshold'])
    learn = float(args['learn'])

    if args['mode'] == 'mode1':
        shape, train_data, test_data = mode1(args['data'], args['ratio'])

    elif args['mode'] == 'mode2':
        shape, train_data, test_data = mode2(args['data'])

    elif args['mode'] == 'mode3':
        shape, train_data, test_data = mode3(args['train'], args['test'])

    network = AdaNetwork(
        'AdaNetwork', shape[_in_], shape[_out_], learn_rate=learn)
    print(network)

    network.train(train_data[_in_], train_data[_out_],
                  max_epoch=epoch, threshold=threshold)

    print("Train score:", network.score(
        train_data[_in_], train_data[_out_]))
    print("Test score:", network.score(
        test_data[_in_], test_data[_out_]))

    if args['mode'] == 'mode3':
        res = network.classify(test_data[_in_])
        with open(args['output'], 'w') as fileout:
            for line in res:
                fileout.write('{}\n'.format(' '.join([str(x) for x in line])))



if __name__ == "__main__":
    main()
