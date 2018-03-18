# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuro.perceptron import Perceptron
from pprint import pprint

################################################################

print()
print('AND')
print()

p = Perceptron('AND', 2, 2, 0.5)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[1, -1], [-1, 1], [-1, 1], [-1, 1]]

p.train(datain, dataout, 1, 1000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Score:', p.score(datain, dataout))
print(64 * '"')

################################################################

print()
print('NAND')
print()

p = Perceptron('NAND', 2, 2, 0.5)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[-1, 1], [-1, 1], [-1, 1], [1, -1]]

p.train(datain, dataout, 1, 1000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Score:', p.score(datain, dataout))
print(64 * '"')

################################################################

print()
print('NOR')
print()

p = Perceptron('NOR', 2, 2, 0.5)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[-1, 1], [1, -1], [1, -1], [1, -1]]

p.train(datain, dataout, 1, 1000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Score:', p.score(datain, dataout))
print(64 * '"')

################################################################

print()
print('XOR')
print()

p = Perceptron('XOR', 2, 2, 0.5)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[1, -1], [-1, 1], [-1, 1], [1, -1]]

p.train(datain, dataout, 1, 1000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Score:', p.score(datain, dataout))
print(64 * '"')
