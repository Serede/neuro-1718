#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, RawTextHelpFormatter

from dataset import Dataset
from metadata import _COURSE_, _YEAR_, _TITLE_, _PAIR_, _AUTHORS_
from per_network import PerNetwork

_input_ = 0
_output_ = 1


def build_parser():
    description = '{} {}\n{}. Perceptrion\nPareja {}: {}'.format(
        _COURSE_, _YEAR_, _TITLE_, _PAIR_, ' y '.join(_AUTHORS_))

    parser = ArgumentParser(description=description,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-t', '--threshold', help='Tolerance for weight updates', required=True)
    parser.add_argument(
        '-l', '--learn', help='Learning rate', required=True)
    parser.add_argument(
        '-e', '--epoch', help='Maximum number of epochs to train', required=True)
    parser.add_argument(
        '-v', '--verbose', help='Display steps', action='store_true')

    return parser


def build_subparser(parser):
    subparsers = parser.add_subparsers(dest='mode', help='Working mode')

    subparsers.required = True

    build_parser1(subparsers)
    build_parser2(subparsers)
    build_parser3(subparsers)


def build_parser1(subparsers):
    parser = subparsers.add_parser(
        'mode1', help='Use one file as dataset split into train and test partitions.')
    parser.add_argument(
        '-d', '--data', help='Dataset file', required=True)
    parser.add_argument(
        '-r', '--ratio', help='Ratio for train (e.g. 0.2 [20%% train/80%% test])', required=True)


def build_parser2(subparsers):
    parser = subparsers.add_parser(
        'mode2', help='Use one file as dataset split into train and test partitions.')
    parser.add_argument(
        '-d', '--data', help='Input dataset file', required=True)


def build_parser3(subparsers):
    parser = subparsers.add_parser(
        'mode3', help='Use one file to train and other file to test.')
    parser.add_argument(
        '-tr', '--train', help='Train dataset file', required=True)
    parser.add_argument(
        '-te', '--test', help='Test dataset file', required=True)


def mode1(data_file, ratio):
    # Load the dataset
    ds = Dataset(data_file)
    shape = ds.input_length, ds.output_length
    train_data, test_data = ds.partition(ratio)

    return shape, train_data, test_data, ds


def mode2(data_file):
    # Load the dataset
    ds = Dataset(data_file)
    shape = ds.input_length, ds.output_length
    train_data = ds.input_data, ds.output_data
    test_data = ds.input_data, ds.output_data

    return shape, train_data, test_data, ds


def mode3(train_file, test_file):
    # Load train dataset
    ds_train = Dataset(train_file)
    shape = ds_train.input_length, ds_train.output_length
    train_data = ds_train.input_data, ds_train.output_data

    # Load test dataset
    ds_test = Dataset(test_file)
    test_data = ds_test.input_data, ds_test.output_data

    return shape, train_data, test_data, ds_train, ds_test


def main():
    size = None
    train_data = None
    test_data = None
    data_set = None
    data_set_test = None
    data_set_train = None

    parser = build_parser()
    build_subparser(parser)

    args = vars(parser.parse_args())

    epoch = float(args['epoch'])
    threshold = float(args['threshold'])
    learn = float(args['learn'])

    print("\n\n","* " * 10)
    if args['mode'] == 'mode1':
        size, train_data, test_data, data_set = mode1(args['data'], float(args['ratio']))

        print("Total instances:",data_set.instance_count)
        print("Train instances:",data_set.train_count)
        print("Test instances:",data_set.test_count)

    elif args['mode'] == 'mode2':
        size, train_data, test_data, data_set = mode2(args['data'])
        print("Total instances:",data_set.instance_count)
        print("Train instances:",data_set.instance_count)
        print("Test instances:",data_set.instance_count)

    elif args['mode'] == 'mode3':
        size, train_data, test_data, data_set_train, data_set_test = mode3(args['train'], args['test'])

        print("Total instances:",data_set_train.instance_count + data_set_test.instance_count)
        print("Train instances:",data_set_train.instance_count)
        print("Test instances:",data_set_test.instance_count)

    network = PerNetwork(name='PerNetwork', input_length=size[_input_], output_length=size[_output_],
                         theta=threshold, learn_rate=learn)
    print(network)

    network.train(train_data[_input_], train_data[_output_], max_epoch=epoch)

    print("Train score:",network.score(train_data[_input_],train_data[_output_]))
    print("Test score:",network.score(test_data[_input_],test_data[_output_]))


if __name__ == "__main__":
    main()
