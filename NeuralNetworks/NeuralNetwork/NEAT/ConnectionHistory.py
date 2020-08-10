from NeuralNetwork.NEAT.Node import node


class connectionHistory:
    def __init__(self, frm: int, to: int, in_no: int, innovationNos: list):
        self.fromNode = frm
        self.toNode = to
        self.innovationNumber = in_no
        self.innovationNumbers = [i for i in innovationNos]

    def matches(self, genome, frm: node, to: node):
        if len(genome.genes) == len(self.innovationNumbers):
            if frm.number == self.fromNode and to.number == self.toNode:
                for i in range(len(genome.genes)):
                    if not genome.genes[i].innovationNo in self.innovationNumbers:
                        return False
                return True
        return False

    def __repr__(self):
        return f"<Connection: from {self.fromNode} to {self.toNode} | Innovation Number : {self.innovationNumber}>"
