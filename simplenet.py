import numpy.matlib
import numpy as np

class Network():
    def __init__(self, layer_lengths):
        # Initialize Nodes
        self.nodes = []
        self.layer_lengths = layer_lengths
        for layer_length in layer_lengths:
            self.nodes.append(np.matlib.zeros((layer_length, 1)))

        # Initialize Weights and Biases
        self.weights = []
        self.biases = []
        for i in range(len(layer_lengths) - 1):
            n = layer_lengths[i]
            k = layer_lengths[i+1]
            self.weights.append(np.matlib.zeros((k, n)))
            self.biases.append(np.matlib.zeros((k, 1)))

    def evaluate(self, starting_nodes):
        if len(starting_nodes) != self.layer_lengths[0]:
            raise Exception('Length of input values passed must be the same as the number of input nodes')
        self.nodes[0] = np.matrix([starting_nodes]).T
        for i in range(1, len(self.nodes)):
            self.nodes[i] = (self.weights[i-1] * self.nodes[i-1]) + self.biases[i-1]
        return self.nodes[len(self.nodes) - 1].T.tolist()[0]


net = Network([2, 1, 3])
net.weights = [np.matrix('2.6 -3'), np.matrix('3; -0.1; 4')]
net.biases = [np.matrix('10'), np.matrix('-2; 6; 12')]
result = net.evaluate([1, 22])
print(result)
