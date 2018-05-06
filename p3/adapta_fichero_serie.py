#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main program.
"""

import sys
from time import time

from seriesadapter import SeriesAdapter

_COURSE_ = 'NEUROCOMPUTACIÓN'
_YEAR_ = '2017-2018'
_TITLE_ = 'Práctica 3: Adapta fichero serie'
_PAIR_ = 'Pareja 08'
_AUTHORS_ = ['Sergio Fuentes', 'Adrián Muñoz']


def main():
    """Main function.
    """
    #./Adapta_fichero_serie fich_entrada fich_salida Na Ns
    if len(sys.argv) != 5:
        raise ValueError('Invalid arguments.\nUsage: Adapta_fichero_serie fich_entrada fich_salida Na Ns')

    na = int(sys.argv[3])
    ns = int(sys.argv[4])
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    series = SeriesAdapter(file_in,na=na,ns=ns)
    series.export(file_out)

    return

if __name__ == "__main__":
    main()
