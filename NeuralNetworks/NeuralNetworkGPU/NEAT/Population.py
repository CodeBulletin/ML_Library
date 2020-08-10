from NeuralNetwork.NEAT.Player import Example
from NeuralNetwork.NEAT.Species import Species
from NeuralNetwork.NEAT.ConnectionHistory import connectionHistory
from NeuralNetwork.utils.defaults import NEAT_default
import math, random, time


class Population:
    def __init__(self, size, player=Example, settings=NEAT_default):
        random.seed(int(time.time()))
        self.size = size
        self.Player = player
        self.settings = settings
        keys = list(NEAT_default.keys())
        SKeys = list(settings.keys())
        for i in keys:
            if i not in SKeys:
                self.settings[i] = NEAT_default[i]
        self.bestScore = 0
        self.gen = 0
        self.massExtinctionEvent = False
        self.newStage = False

        self.pop: list[player] = []
        self.genPlayers: list[player] = []
        self.species: list[Species] = []
        self.innovationHistory: list[connectionHistory] = []

        self.bestPlayer: player = None

        for i in range(self.size):
            self.pop.append(self.Player(self.settings))
            self.pop[i].brain.generateNetwork()
            if not self.settings["networkType"] == "NoConnect":
                if self.settings["networkType"] == "FullyConnect":
                    self.pop[i].brain.fullyConnect(self.innovationHistory)
                elif self.settings["networkType"] == "PartialConnect":
                    self.pop[i].brain.partialConnection(self.innovationHistory)
                elif self.settings["networkType"] == "MinimumConnect":
                    self.pop[i].brain.minimumConnection(self.innovationHistory)
                else:
                    raise TypeError("Incorrect Type")
            self.pop[i].brain.mutate(self.innovationHistory)
            self.pop[i].brain.generateNetwork()

    def done(self):
        for i in self.pop:
            if not i.dead:
                return False
        return True

    def setBestPlayer(self):
        tempBest = self.species[0].players[0]
        tempBest.gen = self.gen
        if tempBest.score > self.bestScore:
            temp = tempBest.cloneForReplay()
            temp.brain.generateNetwork()
            self.genPlayers.append(temp)
            self.bestScore = tempBest.score
            self.bestPlayer = tempBest.cloneForReplay()
            self.bestPlayer.brain.generateNetwork()

    def naturalSelection(self):
        self.speciate()
        self.killEmptySpecies()
        self.calculateFitness()
        self.sortSpecies()
        if self.massExtinctionEvent:
            self.massExtinction()
            self.massExtinctionEvent = False
        self.cullSpecies()
        self.setBestPlayer()
        self.killStaleSpecies()
        self.killBadSpecies()
        averageSum = self.getAvgFitnessSum()
        children = []
        debug = self.settings["debug"]
        if debug:
            print(f"Generation: {self.gen} | Number of mutation: {len(self.innovationHistory)} |"
                  f" Total Species: {len(self.species)}")
        for i in self.species:
            if debug:
                print(f"Best Unadjusted Fitness of Species {self.species.index(i)}: {i.bestFitness}")
                for j in i.players:
                    print(f"<Player: {i.players.index(j)} | Fitness: {j.fitness} | Score: {j.score}>", end=", ")
                print("\n")
            children.append(i.champ.cloneForReplay())
            NoOfChildren = math.floor(i.averageFitness/averageSum * len(self.pop)) - 1
            for _ in range(NoOfChildren):
                children.append(i.giveMeBaby(self.innovationHistory))
        while len(children) < len(self.pop):
            children.append(self.species[0].giveMeBaby(self.innovationHistory))
        self.pop.clear()
        self.pop = children.copy()
        self.gen += 1
        for i in self.pop:
            i.brain.generateNetwork()

    def speciate(self):
        for i in self.species:
            i.players.clear()
        for i in self.pop:
            speciesFound = False
            for j in self.species:
                if j.sameSpecies(i.brain):
                    j.addToSpecies(i)
                    speciesFound = True
                    break
            if not speciesFound:
                self.species.append(Species(p=i, settings=self.settings))

    def killEmptySpecies(self):
        i = 0
        while i < len(self.species):
            if len(self.species[i].players) == 0:
                self.species.remove(self.species[i])
                i -= 1
            i += 1

    def calculateFitness(self):
        for i in self.pop:
            i.calculateFitness()

    def sortSpecies(self):
        for i in self.species:
            i.sortSpecies()
        self.species.sort(key=self.sort_element, reverse=True)

    def massExtinction(self):
        i = 5
        while i < len(self.species):
            self.species.remove(self.species[i])

    def cullSpecies(self):
        for i in self.species:
            i.cull()
            i.fitnessSharing()
            i.setAverage()

    def killStaleSpecies(self):
        i = 2
        while i < len(self.species):
            if self.species[i].staleness >= self.settings["stalenessFactor"]:
                self.species.remove(self.species[i])
                i -= 1
            i += 1

    def killBadSpecies(self):
        averageSum = self.getAvgFitnessSum()
        i = 1
        while i < len(self.species):
            if self.species[i].averageFitness/averageSum * len(self.pop) < 1:
                self.species.remove(self.species[i])
                i -= 1
            i += 1

    def getAvgFitnessSum(self):
        averageSum = 0
        for i in self.species:
            averageSum += i.averageFitness
        return averageSum

    @staticmethod
    def sort_element(x: Species):
        return x.bestFitness
