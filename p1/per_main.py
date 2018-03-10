#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from dataset import Dataset
from per_network import PerNetwork
from metadata import _COURSE_, _YEAR_, _TITLE_, _PAIR_, _AUTHORS_


def build_parser():
    description = '{} {}\n{}. Perceptrón\nPareja {}: {}'.format(
        _COURSE_, _YEAR_, _TITLE_, _PAIR_, ' y '.join(_AUTHORS_))

    parser = ArgumentParser(description=description,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-t', '--theta', help='Threshold', required=True)
    parser.add_argument(
        '-l', '--learn', help='Learning rate', required=True)
    parser.add_argument(
        '-v', '--verbose', help='Display steps', action='store_true')

    return parser


def build_subparser(parser):
    subparsers = parser.add_subparsers(dest='mode', help='Working mode')
    build_parser1(subparsers)
    build_parser2(subparsers)
    build_parser3(subparsers)


def build_parser1(subparsers):
    parser = subparsers.add_parser(
        'mode1', help='1 dataset: random train/test')
    parser.add_argument(
        '-d', '--data', help='Input dataset file', required=True)
    parser.add_argument(
        '-r', '--ratio', help='Percentage for train (e.g. 20 [20%% train/80%% test])', required=True)


def build_parser2(subparsers):
    parser = subparsers.add_parser(
        'mode2', help='1 dataset: full train and test')
    parser.add_argument(
        '-d', '--data', help='Input dataset file', required=True)


def build_parser3(subparsers):
    parser = subparsers.add_parser(
        'mode3', help='2 datasets: train and test')
    parser.add_argument(
        '-tr', '--train', help='Train dataset file', required=True)
    parser.add_argument(
        '-te', '--test', help='Test dataset file', required=True)


def mode1(data_file, ratio):
    # Load the dataset
    ds = Dataset(data_file)
    length = [ds.input_length, ds.output_length]
    [train_data, test_data] = ds.partition(ratio)

    return [length, train_data, test_data]


def mode2(data_file):
    # Load the dataset
    ds = Dataset(data_file)
    length = [ds.input_length, ds.output_length]
    train_data = [ds.input_data, ds.output_data]
    test_data = [ds.input_data, ds.output_data]

    return [length, train_data, test_data]


def mode3(train_file, test_file):
    # Load train dataset
    ds_train = Dataset(train_file)
    length = [ds_train.input_length, ds_train.output_length]
    train_data = [ds_train.input_data, ds_train.output_data]

    # Load test dataset
    ds_test = Dataset(test_file)
    test_data = [ds_test.input_data, ds_test.output_data]

    return [length, train_data, test_data]


def main():

    parser = build_parser()
    build_subparser(parser)

    args = vars(parser.parse_args())

    theta = float(args['theta'])
    learn = float(args['learn'])

    if args['mode'] == 'mode1':
        [length, train_data, test_data] = mode1(args['data'], args['ratio'])

    elif args['mode'] == 'mode2':
        [length, train_data, test_data] = mode2(args['data'])

    elif args['mode'] == 'mode3':
        [length, train_data, test_data] = mode3(args['train'], args['test'])

    network = PerNetwork('PerNetwork', length[0], length[1], theta, learn)
    print(network)

    network.train(train_data[0], train_data[1], verbose=args['verbose'])
    network.run(test_data[0], verbose=args['verbose'])


if __name__ == "__main__":
    main()
