# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple test of logic functions using multilayer perceptrons.
"""

from neuro.ml_perceptron import MLPerceptron
from pprint import pprint

################################################################

print()
print('AND')
print()

p = MLPerceptron('AND', 2, 2, [2], 0.5)
p.randomize_synapses(-0.5, 0.5)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[1, -1], [-1, 1], [-1, 1], [-1, 1]]

p.train(datain, dataout, 1, 1000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Test:')
pprint(p.test(datain))
print('Expected:')
pprint(dataout, width=10)

#print('Score:', p.score(datain, dataout))
print(64 * '"')

################################################################

print()
print('NAND')
print()

p = MLPerceptron('NAND', 2, 2, [2], 0.5)
p.randomize_synapses(-0.5, 0.5)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[-1, 1], [-1, 1], [-1, 1], [1, -1]]

p.train(datain, dataout, 1, 1000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Test:')
pprint(p.test(datain))
print('Expected:')
pprint(dataout, width=10)

#print('Score:', p.score(datain, dataout))
print(64 * '"')

################################################################

print()
print('NOR')
print()

p = MLPerceptron('NOR', 2, 2, [2], 0.5)
p.randomize_synapses(-0.5, 0.5)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[-1, 1], [1, -1], [1, -1], [1, -1]]

p.train(datain, dataout, 1, 1000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Test:')
pprint(p.test(datain))
print('Expected:')
pprint(dataout, width=10)

#print('Score:', p.score(datain, dataout))
print(64 * '"')

################################################################

print()
print('XOR')
print()

p = MLPerceptron('XOR', 2, 2, [2], 0.5)
p.randomize_synapses(-1, 1)
print('Initial synapses:')
pprint(p._synapses)
print()

datain = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
dataout = [[1, -1], [-1, 1], [-1, 1], [1, -1]]

p.train(datain, dataout, 0.25, 10000)
print('Final synapses:')
pprint(p._synapses)
print()

print('Test:')
pprint(p.test(datain))
print('Expected:')
pprint(dataout, width=10)

#print('Score:', p.score(datain, dataout))
print(64 * '"')
