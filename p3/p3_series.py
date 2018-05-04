#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main program.
"""

from time import time
from pprint import pprint

from neuro.parser import Parser, mode1, mode2, mode3
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
        sizein, sizeout, train, test = mode1(args['data'], args['ratio'], shuffle=False)
    elif args['mode'] == 'mode2':
        sizein, sizeout, train, test = mode2(args['data'])
    elif args['mode'] == 'mode3':
        sizein, sizeout, train, test = mode3(args['train'], args['test'])

    p = Series('Series', sizein, sizeout, sizes)
    p.randomize_synapses(-init, init)

    print()
    t0 = time()
    p.train(train[0], train[1], learn, epochs, normalize=normalize)
    t = time()
    print()
    print('Elapsed time: {0:.3f} seconds'.format(t - t0))
    print()

    score, m = p.stats(train[0], train[1])

    print('Train Score:', score)

    if args['mode'] == 'mode1':
        print('Test Score:', p.stats(test[0], test[1])[0])
    elif args['mode'] == 'mode2':
        print('Test Score:', score)
    elif args['mode'] == 'mode3':
        f = args['output']
        res = p.test(test[0])
        with open(f, 'w') as fileout:
            fileout.write('{} {}\n'.format(sizein, sizeout))
            for i in range(len(res)):
                x = ' '.join([str(x) for x in test[0][i]])
                r = [-1] * len(res[i])
                r[res[i].index(max(res[i]))] = 1
                y = ' '.join([str(y) for y in r])
                fileout.write('{} {}\n'.format(x, y))
        print('Test predictions were written to \'{}\'.'.format(f))

    print('Confussion Matrix:')
    prettymatrix(m)


if __name__ == "__main__":
    main()
