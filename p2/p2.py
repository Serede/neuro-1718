#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main program.
"""

from pprint import pprint

from neuro.parser import Parser, mode1, mode2, mode3
from neuro.ml_perceptron import MLPerceptron

_COURSE_ = 'NEUROCOMPUTACIÓN'
_YEAR_ = '2017-2018'
_TITLE_ = 'Práctica 2'
_PAIR_ = 'Pareja 08'
_AUTHORS_ = ['Sergio Fuentes', 'Adrián Muñoz']


def main():
    """Main function.
    """

    parser = Parser(_COURSE_, _YEAR_, _TITLE_, _PAIR_, _AUTHORS_)

    args = vars(parser.parse_args())

    sizes = [int(s) for s in args['sizes']]
    init = float(args['init'])
    learn = float(args['learn'])
    epochs = int(args['epochs'])

    if args['mode'] == 'mode1':
        sizein, sizeout, train, test = mode1(args['data'], args['ratio'])

    elif args['mode'] == 'mode2':
        sizein, sizeout, train, test = mode2(args['data'])

    elif args['mode'] == 'mode3':
        sizein, sizeout, train, test = mode3(args['train'], args['test'])

    p = MLPerceptron('MLPerceptron', sizein, sizeout, sizes)
    p.randomize_synapses(-init, init)

    mse = p.train(train[0], train[1], learn, epochs)

    print('Train Score:', p.score(train[0], train[1], th=0.1))

    if args['mode'] == 'mode3':
        f = args['output']
        res = p.test(test[0])
        with open(f, 'w') as fileout:
            fileout.write('{} {}\n'.format(sizein, sizeout))
            for i in range(len(res)):
                x = ' '.join([str(x) for x in test[0][i]])
                y = ' '.join([str(y) for y in res[i]])
                fileout.write('{} {}\n'.format(x, y))
        print('Test predictions were written to \'{}\'.'.format(f))
    else:
        print('Test Score:', p.score(test[0], test[1], th=0.1))


if __name__ == "__main__":
    main()
