import random
from NeuralNetwork.utils import defaults
from NeuralNetwork.NEAT.Node import node


class connectionGene:
    def __init__(self, frm: node, to: node, w, in_no, settings=defaults.NEAT_default):
        self.fromNode = frm
        self.toNode = to
        self.w = w
        self.innovationNo = in_no
        self.enabled = True
        self.settings = settings
        self.weight_mutation_ratio = self.settings["weight_mutation_ratio"]

    def mutateWeight(self):
        r = random.random()
        if self.weight_mutation_ratio < r:
            self.w = random.uniform(-1, 1)
        else:
            self.w += random.uniform(-1, 1)/50
            if self.w > 1:
                self.w = 1
            elif self.w < -1:
                self.w = -1

    def clone(self, frm: node, to: node):
        cln = connectionGene(frm, to, self.w, self.innovationNo, self.settings)
        cln.enabled = self.enabled
        return cln

    def __repr__(self):
        if self.enabled:
            s = "enabled"
        else:
            s = "disabled"
        ino = self.innovationNo
        return f"<from {self.fromNode.number} to {self.toNode.number} | weight: {self.w} | {s} | Innovation No: {ino}>"
