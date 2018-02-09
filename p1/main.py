#!/usr/bin/env python3

from mp_neuron import MPNeuron
from layer import Layer
from synapse import Synapse
from network import Network

def main():
    inputNeurons = [MPNeuron('input%d' % i, 1) for i in range(3)]
    memoryNeurons = [MPNeuron('memory%d' % i, 1) for i in range(3)]
    detectUpNeurons = [MPNeuron('detectUp%d' % i, 2) for i in range(3)]
    detectDownNeurons = [MPNeuron('detectDown%d' % i, 2) for i in range(3)]
    outputNeurons = [MPNeuron('output%d' % i, 1) for i in range(2)]

    inputLayer = Layer('Input', inputNeurons)
    memoryLayer = Layer('Memory', memoryNeurons)
    detectLayer = Layer('Detect', detectUpNeurons + detectDownNeurons)
    outputLayer = Layer('Output', outputNeurons)

    layers = [inputLayer, memoryLayer, detectLayer, outputLayer]

    memorySynapses = [Synapse(inputNeurons[i], memoryNeurons[i], 1) for i in range(3)]
    inputUpSynapses = [Synapse(inputNeurons[i], detectUpNeurons[i], 1) for i in range(3)]
    inputDownSynapses = [Synapse(inputNeurons[i], detectDownNeurons[i], 1) for i in range(3)]
    memoryUpSynapses = [Synapse(memoryNeurons[(i-1) % 3], detectUpNeurons[i], 1) for i in range(3)]
    memoryDownSynapses = [Synapse(memoryNeurons[(i+1) % 3], detectDownNeurons[i], 1) for i in range(3)]
    outputUpSynapses = [Synapse(detectUpNeurons[i], outputNeurons[0], 1) for i in range(3)]
    outputDownSynapses = [Synapse(detectDownNeurons[i], outputNeurons[1], 1) for i in range(3)]

    synapses = memorySynapses + inputUpSynapses + inputDownSynapses + memoryUpSynapses + memoryDownSynapses + outputUpSynapses + outputDownSynapses

    network = Network('MPNetwork', layers, synapses)

    print(network)

if __name__ == "__main__":
    main()