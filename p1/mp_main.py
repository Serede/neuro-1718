#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from mp_neuron import MPNeuron
from layer import Layer
from synapse import Synapse
from mp_network import MPNetwork
from metadata import _COURSE_, _YEAR_, _TITLE_, _PAIR_, _AUTHORS_


def main():
    description = '{} {}\n{}. McCulloch-Pitts\nPareja {}: {}'.format(
        _COURSE_, _YEAR_, _TITLE_, _PAIR_, ' y '.join(_AUTHORS_))
    parser = ArgumentParser(description=description,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', help='Input file', required=True)
    parser.add_argument(
        '-o', '--output', help='Output file', required=True)
    parser.add_argument(
        '-v', '--verbose', help='Display steps', action='store_true')
    args = vars(parser.parse_args())

    inputNeurons = [MPNeuron('input%d' % i, 1) for i in range(3)]
    memoryNeurons = [MPNeuron('memory%d' % i, 1) for i in range(3)]
    detectUpNeurons = [MPNeuron('detectUp%d' % i, 2) for i in range(3)]
    detectDownNeurons = [MPNeuron('detectDown%d' % i, 2) for i in range(3)]
    outputNeurons = [MPNeuron('output%d' % i, 1) for i in range(2)]

    inputLayer = Layer('Input', inputNeurons)

    outputLayer = Layer('Output', outputNeurons)

    memoryLayer = Layer('Memory', memoryNeurons)
    detectLayer = Layer('Detect', detectUpNeurons + detectDownNeurons)
    hiddenLayers = [memoryLayer, detectLayer]

    memorySynapses = [
        Synapse(inputNeurons[i], memoryNeurons[i], 1) for i in range(3)]
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

    network = MPNetwork(
        'MPNetwork', inputLayer, outputLayer, hiddenLayers, synapses)

    print(network)

    with open(args['input']) as filein:
        datain = [np.asarray(line.split(' ')).astype(float)
                  for line in filein.read().splitlines()]

    dataout = network.run(datain, verbose=args['verbose'])

    with open(args['output'], 'w') as fileout:
        for line in dataout:
            fileout.write('{}\n'.format(line))


if __name__ == "__main__":
    main()
