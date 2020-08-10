from NeuralNetwork.utils.defaults import NEAT_default
from NeuralNetwork.NEAT.Genome import Genome
import random


class Species:
    def __init__(self, p=None, settings=NEAT_default):
        self.players = []
        self.averageFitness = 0
        self.staleness = 0
        self.settings = settings
        keys = list(NEAT_default.keys())
        SKeys = list(settings.keys())
        for i in keys:
            if i not in SKeys:
                self.settings[i] = NEAT_default[i]
        self.excessCoeff = self.settings["excessCoeff"]
        self.weightDiffCoeff = self.settings["weightDiffCoeff"]
        self.compatibilityThreshold = self.settings["compatibilityThreshold"]
        if p is None:
            self.rep = None
            self.champ = None
            self.bestFitness = 0
        else:
            self.players.append(p)
            self.bestFitness = p.fitness
            self.rep = p.brain.clone()
            self.champ = p.cloneForReplay()

    def addToSpecies(self, p):
        self.players.append(p)

    def sortSpecies(self):
        self.players.sort(key=self.sort_element, reverse=True)
        if len(self.players) == 0:
            self.staleness = 200
            return
        if self.players[0].fitness > self.bestFitness:
            self.staleness = 0
            self.bestFitness = self.players[0].fitness
            self.rep = self.players[0].brain.clone()
            self.champ = self.players[0].cloneForReplay()
        else:
            self.staleness += 1

    def setAverage(self):
        Sum = 0
        for i in self.players:
            Sum += i.fitness
        self.averageFitness = Sum / len(self.players)

    def giveMeBaby(self, innovationHistory):
        baby = None
        if random.random() < self.settings["CrossOverPercent"]:
            parent1 = self.SelectPlayer()
            parent2 = self.SelectPlayer()
            if parent1.fitness < parent2.fitness:
                baby = parent2.crossover(parent1)
            else:
                baby = parent1.crossover(parent2)
        else:
            baby = self.SelectPlayer().clone()
        baby.brain.mutate(innovationHistory)
        return baby

    def SelectPlayer(self):
        fitnessSum = 0
        for i in self.players:
            fitnessSum += i.fitness
        rand = random.uniform(0, fitnessSum)
        runningSum = 0
        for i in self.players:
            runningSum += i.fitness
            if runningSum > rand:
                return i
        return self.players[0]

    def sameSpecies(self, g: Genome):
        excessAndDisjoint = self.getExcessDisjoint(g, self.rep)
        averageWeightDiff = self.averageWeightDiff(g, self.rep)
        largeGenomeNormaliser = len(g.genes) - self.settings["largeGenomeNormaliser"]
        if largeGenomeNormaliser < 1:
            largeGenomeNormaliser = 1

        compatibility = (self.excessCoeff * excessAndDisjoint / largeGenomeNormaliser) +\
                        (self.weightDiffCoeff * averageWeightDiff)

        return self.compatibilityThreshold > compatibility

    def cull(self):
        if len(self.players) > 2:
            i = int(len(self.players)/2)
            while i < len(self.players):
                self.players.remove(self.players[i])

    def fitnessSharing(self):
        for i in self.players:
            i.fitness /= len(self.players)

    @staticmethod
    def getExcessDisjoint(brain1: Genome, brain2: Genome):
        matching = 0
        for i in brain1.genes:
            for j in brain2.genes:
                if i.innovationNo == j.innovationNo:
                    matching += 1
                    break
        return len(brain1.genes) + len(brain2.genes) - 2 * matching

    @staticmethod
    def averageWeightDiff(brain1: Genome, brain2: Genome):
        if len(brain1.genes) == 0 or len(brain2.genes) == 0:
            return 0
        matching = 0
        totalDiff = 0
        for i in brain1.genes:
            for j in brain2.genes:
                if i.innovationNo == j.innovationNo:
                    matching += 1
                    totalDiff += abs(i.w - j.w)
                    break
        if matching == 0:
            return 100
        return totalDiff / matching

    @staticmethod
    def sort_element(x):
        return x.fitness
