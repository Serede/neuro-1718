#!/usr/bin/env python3
# -*- coding: utf-8 -*-

_pairno_ = '08'
_authors_ = 'Sergio Fuentes, Adrián Muñoz'

import sys
import numpy as np
import argparse as ap
from mp_neuron import MPNeuron
from layer import Layer
from synapse import Synapse
from mp_network import MPNetwork


def main():
    parser = ap.ArgumentParser(
        description='Práctica 1 - McCulloch-Pitts\nPareja {}: {}'.format(_pairno_, _authors_))
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-v', '--verbose', help='Display network steps',
                        action='store_true', required=False)
    args = vars(parser.parse_args())

    inputNeurons = [MPNeuron('input%d' % i, 1) for i in range(3)]
    memoryNeurons = [MPNeuron('memory%d' % i, 1) for i in range(3)]
    detectUpNeurons = [MPNeuron('detectUp%d' % i, 2) for i in range(3)]
    detectDownNeurons = [MPNeuron('detectDown%d' % i, 2) for i in range(3)]
    outputNeurons = [MPNeuron('output%d' % i, 1) for i in range(2)]

    inputLayer = Layer('Input', inputNeurons)
    memoryLayer = Layer('Memory', memoryNeurons)
    detectLayer = Layer('Detect', detectUpNeurons + detectDownNeurons)
    outputLayer = Layer('Output', outputNeurons)

    hiddenLayers = [memoryLayer, detectLayer]

    memorySynapses = [Synapse(inputNeurons[i], memoryNeurons[i], 1)
                      for i in range(3)]
    inputUpSynapses = [
        Synapse(inputNeurons[i], detectUpNeurons[i], 1) for i in range(3)]
    inputDownSynapses = [
        Synapse(inputNeurons[i], detectDownNeurons[i], 1) for i in range(3)]
    memoryUpSynapses = [
        Synapse(memoryNeurons[(i+1) % 3], detectUpNeurons[i], 1) for i in range(3)]
    memoryDownSynapses = [
        Synapse(memoryNeurons[(i-1) % 3], detectDownNeurons[i], 1) for i in range(3)]
    outputUpSynapses = [
        Synapse(detectUpNeurons[i], outputNeurons[0], 1) for i in range(3)]
    outputDownSynapses = [
        Synapse(detectDownNeurons[i], outputNeurons[1], 1) for i in range(3)]

    synapses = memorySynapses + inputUpSynapses + inputDownSynapses + \
        memoryUpSynapses + memoryDownSynapses + outputUpSynapses + outputDownSynapses

    network = MPNetwork('MPNetwork', inputLayer,
                        outputLayer, hiddenLayers, synapses)

    print(network)

    with open(args['input']) as filein, open(args['output'], 'w') as fileout:
        datain = [np.asarray(line.split(' ')).astype(float)
                  for line in filein.read().splitlines()]
        dataout = network.run(datain, verbose=args['verbose'])
        for line in dataout:
            fileout.write('{}\n'.format(line))


if __name__ == "__main__":
    main()
