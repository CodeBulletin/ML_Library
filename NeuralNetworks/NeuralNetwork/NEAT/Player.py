"""\
Class Variables:
    fitness: Fitness of Player
    settings: Settings
    genomeInputs: Number of inputs to Genome
    genomeOutputs: Number of outputs to Genome
    brain: Brain of Player
    score: Score of The Player
    gen: Generation of Player

Class Functions:
    clone: To create Clone
    cloneForReplay: To create Clone for replay
    calculateFitness: To calculate fitness
    crossover: To get the child using crossover

How to Write class for this library
```
class Example(Player):
    def __init__(self, settings):
        super().__init__(settings)

    def clone(self):
        clone = Example(self.settings)
        clone.brain = self.brain.clone()
        clone.fitness = self.fitness
        clone.brain.generateNetwork()
        clone.gen = self.gen
        clone.bestScore = self.score
        return clone

    def cloneForReplay(self):
        clone = Example(self.settings)
        clone.brain = self.brain.clone()
        clone.fitness = self.fitness
        clone.brain.generateNetwork()
        clone.gen = self.gen
        clone.bestScore = self.score
        #The Things you want to do
        return clone

    def crossover(self, parent2):
        child = Example(self.settings)
        child.brain = self.brain.crossover(parent2.brain)
        child.brain.generateNetwork()
        return child

    def calculateFitness(self):
        self.fitness = 100/(len(self.brain.genes)+1)
```
"""
from NeuralNetwork.NEAT.Genome import Genome
from NeuralNetwork.utils.defaults import NEAT_default
import abc


class Player(abc.ABC):
    def __init__(self, settings=NEAT_default):
        self.fitness = 0
        self.settings = settings
        self.genomeInputs = settings["genomeInputs"]
        self.genomeOutputs = settings["genomeOutputs"]
        self.brain = Genome(self.genomeInputs, self.genomeOutputs, settings=settings)
        self.score = 0
        self.gen = 0
        self.vision = []

    @abc.abstractmethod
    def clone(self):
        pass

    @abc.abstractmethod
    def cloneForReplay(self):
        pass

    @abc.abstractmethod
    def calculateFitness(self):
        pass

    @abc.abstractmethod
    def crossover(self, parent2):
        pass


class Example(Player):
    def __init__(self, settings):
        super().__init__(settings)

    def clone(self):
        clone = Example(self.settings)
        clone.brain = self.brain.clone()
        clone.fitness = self.fitness
        clone.brain.generateNetwork()
        clone.gen = self.gen
        clone.bestScore = self.score
        return clone

    def cloneForReplay(self):
        clone = Example(self.settings)
        clone.brain = self.brain.clone()
        clone.fitness = self.fitness
        clone.brain.generateNetwork()
        clone.gen = self.gen
        clone.bestScore = self.score
        return clone

    def crossover(self, parent2):
        child = Example(self.settings)
        child.brain = self.brain.crossover(parent2.brain)
        child.brain.generateNetwork()
        return child

    def calculateFitness(self):
        self.fitness = 100/(len(self.brain.genes)+1)
