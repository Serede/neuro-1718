#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main program for Series.
"""

from time import time
from pprint import pprint

from neuro.parser import Parser, mode1, mode2, mode3, modeR
from neuro.series import Series

_COURSE_ = 'NEUROCOMPUTACIÓN'
_YEAR_ = '2017-2018'
_TITLE_ = 'Práctica 2'
_PAIR_ = 'Pareja 08'
_AUTHORS_ = ['Sergio Fuentes', 'Adrián Muñoz']


def prettymatrix(m):
    """Pretty print matrix.

    Args:
        m (list): Matrix.
    """

    s = [[str(e) for e in row] for row in m]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    tbl = [fmt.format(*row) for row in s]
    print('\n'.join(tbl))


def main():
    """Main function.
    """

    parser = Parser(_COURSE_, _YEAR_, _TITLE_, _PAIR_, _AUTHORS_)

    args = vars(parser.parse_args())

    sizes = [int(s) for s in args['sizes']]
    init = float(args['init'])
    learn = float(args['learn'])
    epochs = int(args['epochs'])
    normalize = args['normalize']

    if args['mode'] == 'mode1':
        sizein, sizeout, train, test = mode1(args['data'], args['ratio'])
    elif args['mode'] == 'mode2':
        sizein, sizeout, train, test = mode2(args['data'])
    elif args['mode'] == 'mode3':
        sizein, sizeout, train, test = mode3(args['train'], args['test'])
    elif args['mode'] == 'modeR':
        sizein, sizeout, train, test = modeR(args['train'], args['proportion'])

        p = Series('Series', sizein, sizeout, sizes)
        p.randomize_synapses(-init, init)

        n_recursive = int(args['n_epochs'])
        p.train(train[0], train[1], learn, epochs, normalize=False)
        prediction_r, prediction = p.predict_recursive(test[0], n_recursive)

        # Write the results to a file
        # in the second line the respective prediction
        # in the third line the complete prediction
        with open(args['output'], 'w') as file_out:
            # in the first line. the test output
            file_out.writelines(' '.join([str(e[0]) for e in test[1]]))
            file_out.writelines('\n')
            # in the second line the recursive prediction
            file_out.writelines(' '.join([str(p) for p in prediction_r]))
            file_out.writelines('\n')
            # in the third line the std prediction
            file_out.writelines(' '.join([str(p[0]) for p in prediction]))
            file_out.writelines('\n')
        return

    p = Series('Series', sizein, sizeout, sizes)
    p.randomize_synapses(-init, init)

    print()
    t0 = time()
    p.train(train[0], train[1], learn, epochs, normalize=normalize)
    t = time()
    print()
    print('Elapsed time: {0:.3f} seconds'.format(t - t0))
    print()


    ecm, basic = p.stats(test[0], test[1])

    print('Statistics:')
    print('ECM = ', ecm)
    print('Basic ECM = ', basic)

    if args['mode'] == 'mode3':
        with open(args['output'], 'w') as file_out:
            results = p.test(test[0])
            for r in results:
                for x in r:
                    file_out.write(str(x) + '\n')


if __name__ == "__main__":
    main()
