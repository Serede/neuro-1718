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

def parser_mode_1(subparsers):
    parser1 = subparsers.add_parser(
        'mode1', help='1 dataset: random train/test')
    parser1.add_argument(
        '-d', '--data', help='Input dataset file', required=True)
    parser1.add_argument(
        '-r', '--ratio', help='Percentage for train (e.g. 20 [20%% train/80%% test])', required=True)

    return parser1

def parser_mode_2(subparsers):
    parser2 = subparsers.add_parser(
        'mode2', help='1 dataset: full train and test')
    parser2.add_argument(
        '-d', '--data', help='Input dataset file', required=True)

    return parser2

def parser_mode_3(subparsers):
    parser3 = subparsers.add_parser(
        'mode3', help='2 datasets: train and test')
    parser3.add_argument(
        '-tr', '--train', help='Train dataset file', required=True)
    parser3.add_argument(
        '-te', '--test', help='Test dataset file', required=True)

def mode_2(args):
    # Load the dataset
    ds = Dataset(args['data'])
    len_in = ds.input_length
    len_out = ds.output_length

    train_input = ds.input_data
    train_output = ds.output_data

    test_input = ds.input_data
    test_output = ds.output_data


    network = PerNetwork('PerNetwork', len_in, len_out, float(args['theta']), float(args['learn']))

    network.train(train_input,train_output)
    network.score(test_input,test_output)


def main():

    parser = build_parser()

    subparsers = parser.add_subparsers(dest='mode', help='Working mode')

    parser1 = parser_mode_1(subparsers)
    parser2 = parser_mode_2(subparsers)
    parser3 = parser_mode_3(subparsers)

    args = vars(parser.parse_args())

    if args['mode'] == 'mode1':

        ds = Dataset(args['data'])
        len_in = ds.input_length
        len_out = ds.output_length
        [train_data, test_data] = ds.partition(args['ratio'])

    elif args['mode'] == 'mode2':

        mode_2(args)


    elif args['mode'] == 'mode3':

        ds_train = Dataset(args['train'])
        len_in = ds_train.input_length
        len_out = ds_train.output_length
        train_data = [ds_train.input_data, ds_train.output_data]
        ds_test = Dataset(args['test'])
        test_data = ds_test.input_data

    network = PerNetwork('PerNetwork', len_in,
                         len_out, args['theta'], args['learn'])
    print(network)

    network.train(train_data[0], train_data[1], verbose=args['verbose'])
    network.run(test_data, verbose=args['verbose'])


if __name__ == "__main__":
    main()
