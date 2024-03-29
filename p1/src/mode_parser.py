#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.dataset import Dataset


def build_mode_subparser(parser):
    subparsers = parser.add_subparsers(dest='mode', help='Working mode')

    subparsers.required = True

    build_mode1_parser(subparsers)
    build_mode2_parser(subparsers)
    build_mode3_parser(subparsers)


def build_mode1_parser(subparsers):
    parser = subparsers.add_parser(
        'mode1', help='Use one file as dataset split into train and test partitions.')
    parser.add_argument(
        '-d', '--data', help='Dataset file', required=True)
    parser.add_argument(
        '-r', '--ratio', help='Ratio for train (e.g. 0.2 [20%% train/80%% test])', required=True)


def build_mode2_parser(subparsers):
    parser = subparsers.add_parser(
        'mode2', help='Use one file as dataset split into train and test partitions.')
    parser.add_argument(
        '-d', '--data', help='Input dataset file', required=True)


def build_mode3_parser(subparsers):
    parser = subparsers.add_parser(
        'mode3', help='Use one file to train and other file to test.')
    parser.add_argument(
        '-tr', '--train', help='Train dataset file', required=True)
    parser.add_argument(
        '-te', '--test', help='Test dataset file', required=True)
    parser.add_argument(
        '-o', '--output', help='Output file for predictions', required=True)


def mode1(data_file, ratio):
    print('RUNNING IN MODE 1')

    # Load the dataset
    ds = Dataset(data_file)
    shape = ds.input_length, ds.output_length
    train_data, test_data = ds.partition(float(ratio))
    print('Train instances: {}'.format(len(train_data[0])))
    print('Test instances: {}'.format(len(test_data[0])))

    return shape, train_data, test_data


def mode2(data_file):
    print('RUNNING IN MODE 2')

    # Load the dataset
    ds = Dataset(data_file)
    shape = ds.input_length, ds.output_length
    train_data = ds.input_data, ds.output_data
    test_data = ds.input_data, ds.output_data
    print('Train instances: {}'.format(len(train_data[0])))
    print('Test instances: {}'.format(len(test_data[0])))

    return shape, train_data, test_data


def mode3(train_file, test_file):
    print('RUNNING IN MODE 3')

    # Load train dataset
    ds_train = Dataset(train_file)
    shape = ds_train.input_length, ds_train.output_length
    train_data = ds_train.input_data, ds_train.output_data

    # Load test dataset
    ds_test = Dataset(test_file)
    test_data = ds_test.input_data, ds_test.output_data

    print('Train instances: {}'.format(len(train_data[0])))
    print('Test instances: {}'.format(len(test_data[0])))

    return shape, train_data, test_data
