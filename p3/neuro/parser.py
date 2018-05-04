# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument parser and running modes.
"""

from argparse import ArgumentParser, RawTextHelpFormatter

from neuro.dataset import Dataset


class Parser():
    """Parser class.

    Attributes:
        course (str): Course name.
        year (str): Academic year.
        title (str): Program title.
        team (str): Team information.
        authors (list): List of authors.
    """

    course = None
    year = None
    title = None
    team = None
    authors = None

    _parser = None

    def __init__(self, course, year, title, team, authors):
        self.course = course
        self.year = year
        self.title = title
        self.team = team
        self.authors = authors

        # Create parser
        desc = (
            '{} {}\n'
            '{}\n'
            '{}: {}'
        ).format(course, year, title, team, ', '.join(authors))
        self._parser = ArgumentParser(description=desc,
                                      formatter_class=RawTextHelpFormatter)
        self._parser.add_argument(
            '-s', '--sizes', help='list of sizes for hidden layers', required=True, nargs='+')
        self._parser.add_argument(
            '-i', '--init', help='weight initialization boundary (e.g. \'1\' => [-1, 1])', required=True)
        self._parser.add_argument(
            '-l', '--learn', help='learning rate', required=True)
        self._parser.add_argument(
            '-e', '--epochs', help='maximum number of training epochs', required=True)
        self._parser.add_argument(
            '-z', '--normalize', help='normalize data', action='store_true')

        # Create subparsers
        sp = self._parser.add_subparsers(dest='mode', help='Working mode')
        sp.required = True

        # Mode 1 subparser
        p = sp.add_parser(
            'mode1', help='1 file. Random partitions for train and test.')
        p.add_argument('-d', '--data', help='Dataset file', required=True)
        p.add_argument(
            '-r', '--ratio', help='Ratio for train (e.g. \'0.2\' => 20%% train, 80%% test)', required=True)

        # Mode 2 subparser
        parser = sp.add_parser(
            'mode2', help='1 file. Full partition for both train and test.')
        parser.add_argument('-d', '--data', help='Dataset file', required=True)

        # Mode 3 subparser
        parser = sp.add_parser(
            'mode3', help='2 files. Separate partitions for train and test.')
        parser.add_argument(
            '-d', '--train', help='Train dataset file', required=True)
        parser.add_argument(
            '-t', '--test', help='Test dataset file', required=True)
        parser.add_argument(
            '-o', '--output', help='Output file for predictions', required=True)

        self.parse_args = self._parser.parse_args

    parse_args = None


def mode1(data_file, ratio, shuffle=True):
    """Prepares data for mode 1.

    Args:
        data_file (str): Dataset file.
        ratio (float): Ratio for train.
        shuffle (bool, optional): Defaults to True. Shuffle data.

    Returns:
        tuple: (sizein, sizeout, train, test)
    """

    print('Running in Mode 1')
    # Load the dataset
    ds = Dataset(data_file)
    train, test = ds.partition(float(ratio), shuffle=shuffle)
    print('- Train instances: {}'.format(len(train[0])))
    print('- Test instances: {}'.format(len(test[0])))

    return ds.sizein, ds.sizeout, train, test


def mode2(data_file,shuffle=True):
    """Prepares data for mode 2.

    Args:
        data_file (str): Dataset file.

    Returns:
        tuple: (sizein, sizeout, train, test)
    """

    print('Running in Mode 2')
    # Load the dataset
    ds = Dataset(data_file)
    train = ds.data()
    test = ds.data()
    print('- Train instances: {}'.format(len(train[0])))
    print('- Test instances: {}'.format(len(test[0])))

    return ds.sizein, ds.sizeout, train, test


def mode3(train_file, test_file):
    """Prepares data for mode 3.

    Args:
        train_file (str): Train dataset file.
        test_file (str): Test dataset file.

    Returns:
        tuple: (sizein, sizeout, train, test)
    """
    print('Running in Mode 3')
    # Load train dataset
    ds1 = Dataset(train_file)
    train = ds1.data()
    # Load test dataset
    ds2 = Dataset(test_file)
    test = ds2.data()

    print('- Train instances: {}'.format(len(train[0])))
    print('- Test instances: {}'.format(len(test[0])))

    return ds1.sizein, ds1.sizeout, train, test
