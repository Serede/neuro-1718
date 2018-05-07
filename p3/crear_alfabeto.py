#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main program for Crear_alfabeto.
"""

import sys

from alphabet import Alphabet

_COURSE_ = 'NEUROCOMPUTACIÓN'
_YEAR_ = '2017-2018'
_TITLE_ = 'Práctica 3: Crear Alfabeto'
_PAIR_ = 'Pareja 08'
_AUTHORS_ = ['Sergio Fuentes', 'Adrián Muñoz']


def main():
    """Main function.
    """
    # ./ Crear_alfabeto num_copias num_errores fich_entrada fich_salida
    if len(sys.argv) != 5:
        raise ValueError(
            'Invalid arguments.\nUsage: Crear_alfabeto num_copias num_errores fich_entrada fich_salida')

    n_copies = int(sys.argv[1])
    n_errors = int(sys.argv[2])
    file_in = sys.argv[3]
    file_out = sys.argv[4]

    a = Alphabet(filename=file_in)
    a.export(n_copies, n_errors, filename=file_out)

    return


if __name__ == "__main__":
    main()
