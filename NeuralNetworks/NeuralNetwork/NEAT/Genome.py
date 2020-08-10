from NeuralNetwork.utils.defaults import NEAT_default
from NeuralNetwork.NEAT.Node import node
from NeuralNetwork.NEAT.ConnectionGene import connectionGene
from NeuralNetwork.NEAT.ConnectionHistory import connectionHistory
import math, random


class Genome:
    def __init__(self, input_nodes, output_nodes, settings=NEAT_default, b=False):
        self.input_node = input_nodes
        self.output_node = output_nodes
        self.nodes: list[node] = []
        self.genes: list[connectionGene] = []
        self.network = []
        self.settings = settings
        keys = list(NEAT_default.keys())
        SKeys = list(settings.keys())
        for i in keys:
            if i not in SKeys:
                self.settings[i] = NEAT_default[i]
        self.nextNode = 0
        self.layers = 2
        self.biasNode = 0
        if not b:
            for i in range(input_nodes):
                self.nodes.append(node(i))
                self.nextNode += 1
            for i in range(output_nodes):
                self.nodes.append(node(i+input_nodes, settings["output_activation"]))
                self.nodes[i+input_nodes].layer = 1
                self.nextNode += 1
            self.nodes.append(node(self.nextNode))
            self.biasNode = self.nextNode
            self.nextNode += 1

    def fullyConnect(self, innovationHistory):
        for i in range((self.input_node+1)*self.output_node):
            self.addConnection(innovationHistory)
        self.connectNodes()

    def partialConnection(self, innovationHistory):
        for i in range(int(round(((self.input_node+1)*self.output_node)/2))):
            self.addConnection(innovationHistory)
        self.connectNodes()

    def minimumConnection(self, innovationHistory):
        for i in range(3):
            self.addConnection(innovationHistory)
        self.connectNodes()

    def getNode(self, node_number):
        for i in self.nodes:
            if i.number == node_number:
                return i

    def connectNodes(self):
        for i in self.nodes:
            i.output_connections.clear()
        for i in self.genes:
            i.fromNode.output_connections.append(i)

    def feedForward(self, input_values):
        for i in range(self.input_node):
            self.nodes[i].outputValue = input_values[i]
        self.nodes[self.biasNode].outputValue = 1
        for i in range(len(self.network)):
            self.network[i].engage()
        outs = []
        for i in range(self.output_node):
            outs.append(self.nodes[self.input_node + i].outputValue)
        for i in range(len(self.nodes)):
            self.nodes[i].input_sum = 0
        return outs

    def generateNetwork(self):
        self.connectNodes()
        self.network = []
        for i in range(self.layers):
            for j in self.nodes:
                if j.layer == i:
                    self.network.append(j)

    def addNode(self, innovationHistory):
        if len(self.genes) == 0:
            self.addConnection(innovationHistory)
            return
        rc = int(math.floor(random.uniform(0, len(self.genes))))
        while self.genes[rc].fromNode == self.nodes[self.biasNode] and len(self.genes) != 1:
            rc = int(math.floor(random.uniform(0, len(self.genes))))
        self.genes[rc].enabled = False
        newNodeNo = self.nextNode
        self.nodes.append(node(newNodeNo, random.choice(self.settings["hidden_activation_list"])))
        self.nextNode += 1
        gN = self.getNode(newNodeNo)
        CNI = self.getInnovationNumber(innovationHistory, self.genes[rc].fromNode, gN)
        self.genes.append(connectionGene(self.genes[rc].fromNode, gN, 1, CNI, self.settings))
        CNI = self.getInnovationNumber(innovationHistory, gN, self.genes[rc].toNode)
        self.genes.append(connectionGene(gN, self.genes[rc].toNode, self.genes[rc].w, CNI, self.settings))
        gN.layer = self.genes[rc].fromNode.layer + 1
        CNI = self.getInnovationNumber(innovationHistory, self.nodes[self.biasNode], gN)
        self.genes.append(connectionGene(self.nodes[self.biasNode], gN, 0, CNI, self.settings))
        if gN.layer == self.genes[rc].toNode.layer:
            for i in range(len(self.nodes)-1):
                if self.nodes[i].layer >= gN.layer:
                    self.nodes[i].layer += 1
            self.layers += 1
        self.connectNodes()

    def addConnection(self, innovationHistory):
        if self.fullyConnected():
            return
        randomNode1 = int(math.floor(random.uniform(0, len(self.nodes))))
        randomNode2 = int(math.floor(random.uniform(0, len(self.nodes))))
        while self.randomConnectionsAreBad(randomNode1, randomNode2):
            randomNode1 = int(math.floor(random.uniform(0, len(self.nodes))))
            randomNode2 = int(math.floor(random.uniform(0, len(self.nodes))))
        if self.nodes[randomNode1].layer > self.nodes[randomNode2].layer:
            randomNode1, randomNode2 = randomNode2, randomNode1
        CIN = self.getInnovationNumber(innovationHistory, self.nodes[randomNode1], self.nodes[randomNode2])
        temp = connectionGene(self.nodes[randomNode1], self.nodes[randomNode2],
                              random.uniform(1, -1), CIN, self.settings)
        self.genes.append(temp)
        self.connectNodes()

    def getInnovationNumber(self, innovationHistory, frm: node, to: node):
        isNew = True
        connectionInnovationNumber = self.settings["__nextConnectionNo"]
        for i in innovationHistory:
            if i.matches(self, frm, to):
                isNew = False
                connectionInnovationNumber = i.innovationNumber
                break

        if isNew:
            in_nos = []
            for i in self.genes:
                in_nos.append(i.innovationNo)
            innovationHistory.append(connectionHistory(frm.number, to.number, connectionInnovationNumber, in_nos))
            self.settings["__nextConnectionNo"] += 1
        return connectionInnovationNumber

    def fullyConnected(self):
        maxConnections = 0
        nodes_in_layers = [0 for _ in range(self.layers)]
        for i in self.nodes:
            nodes_in_layers[i.layer] += 1
        for i in range(self.layers - 1):
            nodes_in_front = 0
            for j in range(i+1, self.layers):
                nodes_in_front += nodes_in_layers[j]
            maxConnections += nodes_in_layers[i]*nodes_in_front
        if maxConnections == len(self.genes):
            return True
        return False

    def randomConnectionsAreBad(self, a, b):
        if self.nodes[a].layer == self.nodes[b].layer:
            return True
        if self.nodes[a].IsConnectedTo(self.nodes[b]):
            return True
        return False

    def crossover(self, parent2):
        child = Genome(self.input_node, self.output_node, self.settings, True)
        child.genes.clear()
        child.nodes.clear()
        child.layers = self.layers
        child.nextNode = self.nextNode
        child.biasNode = self.biasNode
        childGene = []
        isEnabled = []
        for i in range(len(self.genes)):
            setEnabled = True
            parent2Gene = self.matchingGene(parent2, self.genes[i].innovationNo)
            if parent2Gene != -1:
                if (not self.genes[i].enabled) or (not parent2.genes[i].enabled):
                    if random.random() < self.settings["EnablePercent"]:
                        setEnabled = False
                if random.random() < self.settings["parentGenePercent"]:
                    childGene.append(self.genes[i])
                else:
                    childGene.append(parent2.genes[parent2Gene])
            else:
                childGene.append(self.genes[i])
                setEnabled = self.genes[i].enabled
            isEnabled.append(setEnabled)
        for i in self.nodes:
            child.nodes.append(i.clone())
        for i in range(len(childGene)):
            child.genes.append(childGene[i].clone(child.getNode(childGene[i].fromNode.number),
                                                  child.getNode(childGene[i].toNode.number)))
            child.genes[i].enabled = isEnabled[i]
        child.connectNodes()
        return child

    def mutate(self, innovationHistory):
        if random.random() < self.settings["WeightMPercent"]:
            for i in self.genes:
                i.mutateWeight()
        if random.random() < self.settings["ConnectionMPercent"]:
            self.addConnection(innovationHistory)
        if random.random() < self.settings["NodeMPercent"]:
            self.addNode(innovationHistory)

    def clone(self):
        clone = Genome(self.input_node, self.output_node, settings=self.settings, b=True)
        for i in self.nodes:
            clone.nodes.append(i.clone())
        for i in range(len(self.genes)):
            clone.genes.append(self.genes[i].clone(clone.getNode(self.genes[i].fromNode.number),
                                                   clone.getNode(self.genes[i].toNode.number)))
        clone.layers = self.layers
        clone.nextNode = self.nextNode
        clone.biasNode = self.biasNode
        clone.connectNodes()
        return clone

    @staticmethod
    def matchingGene(parent2, innovationNumber):
        for i in range(len(parent2.genes)):
            if parent2.genes[i].innovationNo == innovationNumber:
                return i
        return -1
