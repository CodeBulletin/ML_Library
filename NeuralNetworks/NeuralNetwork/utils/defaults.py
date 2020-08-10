TanH = "TanH"
Sigmoid = "Sigmoid"
ReLu = "ReLu"
LReLu = "LeakyReLu"
PReLu = "ParametrizedReLu"
ELU = "ExponentialLU"
SoftPlus = "SoftPlus"
SoftMax = "SoftMax"
LinearActivation = "LinearActivation"

Momentum = "Momentum"
RMSprop = "RMSprop"
SGD = "SGD"
Adam = "Adam"

L2 = "L2"

CrossEntropy = "CrossEntropy"
MSE = "MSE"
MAE = "MAE"

LinearScaling = "LinearScaling"
LogScaling = "LogScaling"
ZScore = "z_score"

default_settings = {
    "cost": MSE,
    "regularization": L2,
    "regularization_lambda": 2,
    "optimizer": Adam,
    "normalizer": LinearScaling
}

FullyConnect = "FullyConnect"
PartialConnect = "PartialConnect"
MinimumConnect = "MinimumConnect"
NoConnect = "NoConnect"

NEAT_default = {
    "hidden_activation_list": [TanH],
    "output_activation": TanH,
    "genomeInputs": 3,
    "genomeOutputs": 3,
    "weight_mutation_ratio": 0.01,
    "EnablePercent": 0.75,
    "parentGenePercent": 0.5,
    "WeightMPercent": 0.8,
    "ConnectionMPercent": 0.08,
    "NodeMPercent": 0.02,
    "CrossOverPercent": 0.75,
    "excessCoeff": 1.0,
    "weightDiffCoeff": 0.5,
    "compatibilityThreshold": 3.0,
    "largeGenomeNormaliser": 20.0,
    "stalenessFactor": 15,
    "debug": False,
    "networkType": NoConnect,
    "__nextConnectionNo": 1000
}
