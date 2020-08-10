from NeuralNetwork.utils import Activation
from NeuralNetwork.utils import defaults


Activations = {
    "TanH": Activation.TanH().fn,
    "Sigmoid": Activation.Sigmoid().fn,
    "ReLu": Activation.ReLu().fn,
    "LeakyReLu": Activation.LReLU().fn,
    "SoftPlus": Activation.SoftPlus().fn,
    "LinearActivation": Activation.LinearActivation().fn
}


class node:
    def __init__(self, ino, activation_name=defaults.Sigmoid):
        self.number = ino
        self.input_sum = 0
        self.outputValue = 0
        self.activation_name = activation_name
        self.activation = Activations[self.activation_name]
        self.output_connections = []
        self.layer = 0

    def engage(self):
        if self.layer != 0:
            self.outputValue = self.activation(self.input_sum)
        for i in range(len(self.output_connections)):
            # print(self.number)
            if self.output_connections[i].enabled:
                self.output_connections[i].toNode.input_sum += self.output_connections[i].w*self.outputValue

    def clone(self):
        cln = node(self.number, self.activation_name)
        cln.layer = self.layer
        return cln

    def IsConnectedTo(self, other):
        if other.layer == self.layer:
            return False
        elif other.layer < self.layer:
            for i in range(len(other.output_connections)):
                if other.output_connections[i].toNode == self:
                    return True
        else:
            for i in range(len(self.output_connections)):
                if self.output_connections[i].toNode == other:
                    return True
        return False
