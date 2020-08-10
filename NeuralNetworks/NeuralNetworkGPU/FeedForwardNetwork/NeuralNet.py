import cupy as cp
from NeuralNetwork.utils import Activation, cost, regularization, optimizers, Normalizer
from NeuralNetwork.utils.defaults import TanH, SoftMax, default_settings
import json


class NeuralNetwork(optimizers.optimizers):
    def __init__(self, input_neurons=0, network_shape=None, learning_rate=0.1, beta=0.9, gamma=0.999, epsilon=1e-8,
                 activation=TanH, settings=None, activation_param=None,
                 do_regularization=False, do_normalization=True):
        super().__init__()
        if network_shape is None:
            network_shape = []
        self.ActivationList = {}
        self.RegularizationList = {}
        self.CostList = {}
        self.OptimizerList = {}
        self.NormList = {}
        self.Activations = []
        self.param = activation_param
        self.make_dict()
        self.shape = [input_neurons]
        self.activation = []
        self.derivative = []
        self.make(network_shape, activation)
        self.learning_rate = learning_rate
        self.weights_shape = []
        self.weights = []
        self.biases = []
        self.cost_key = default_settings["cost"]
        self.cost = None
        self.d_cost = None
        self.regularization_key = None
        self.regularization = None
        self.lam = None
        self.do_regularization = do_regularization
        self.optimizer_key = default_settings["optimizer"]
        self.optimizer = self.OptimizerList[self.optimizer_key]
        self.do_norm = do_normalization
        self.norm_key = default_settings["normalizer"]
        self.normalizer = self.NormList[self.norm_key]
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        if settings is None:
            self.settings = default_settings
        else:
            self.settings = settings
        self.set_settings()
        for i in range(len(self.shape) - 1):
            self.weights_shape.append([self.shape[i + 1], self.shape[i]])
            self.weights.append(
                cp.random.standard_normal(self.weights_shape[i]) / (self.weights_shape[i][1] ** 0.5))
            self.biases.append(cp.random.standard_normal([self.shape[i + 1], 1]))

    def copy(self):
        model = NeuralNetwork()
        model.shape = [i for i in self.shape]
        model.weights_shape = [[i for i in j] for j in self.weights_shape]
        model.learning_rate = self.learning_rate
        model.Activations = [i for i in self.Activations]
        model.param = self.param
        model.optimizer_key = self.optimizer_key
        model.cost_key = self.cost_key
        model.beta = self.beta
        model.gamma = self.gamma
        model.epsilon = self.epsilon
        model.do_regularization = self.do_regularization
        model.regularization_key = self.regularization_key
        model.regularization = self.regularization
        model.do_norm = self.do_norm
        model.norm_key = self.norm_key
        model.weights = [i.copy() for i in self.weights]
        model.biases = [i.copy() for i in self.biases]
        model.activation = [model.ActivationList[i].fn for i in model.Activations]
        model.derivative = [model.ActivationList[i].d for i in model.Activations]
        model.optimizer = model.OptimizerList[model.optimizer_key]
        model.cost = model.CostList[model.cost_key].cost
        model.d_cost = model.CostList[model.cost_key].d_cost
        model.normalizer = model.NormList[model.norm_key]
        return model

    def mutate(self, mutation_rate, mutation_ratio):
        def internal(x, mr, mro, element):
            temp1 = cp.random.uniform(0, 1)
            if temp1 < mr:
                temp2 = cp.random.uniform()
                if temp2 < mro:
                    return cp.random.standard_normal() / (self.weights_shape[element][1] ** 0.5)
                else:
                    return x + cp.random.uniform(-x / 100, x / 100)
            else:
                return x
        randomize = cp.vectorize(internal)
        for i in range(len(self.weights)):
            self.weights[i] = randomize(self.weights[i], mutation_rate, mutation_ratio, i)
            self.biases[i] = randomize(self.biases[i], mutation_rate, mutation_ratio, i)

    def crossover(self, parent2):
        child = self.copy()
        for i in range(len(self.weights)):
            r = cp.random.uniform(0, 1)
            if r < 0.5:
                pass
            else:
                child.weights[i] = cp.copy(parent2.weights[i])
                child.biases[i] = cp.copy(parent2.biases[i])
        return child

    def set_settings(self):
        keys = list(self.settings.keys())
        if "d_cost" not in keys:
            if "cost" in keys:
                self.cost_key = self.settings["cost"]
                self.cost = self.CostList[self.cost_key].cost
                self.d_cost = self.CostList[self.cost_key].d_cost
            else:
                self.cost = self.CostList[self.cost_key].cost
                self.d_cost = self.CostList[self.cost_key].d_cost
        else:
            self.cost = self.settings["cost"]
            self.d_cost = self.settings["d_cost"]
        if self.do_regularization:
            if "regularization" in keys:
                self.regularization_key = self.settings["regularization"]
                self.regularization = self.RegularizationList[self.regularization_key]
            else:
                self.regularization_key = default_settings["regularization"]
                self.regularization = self.RegularizationList[self.regularization_key]
            if "regularization_lambda" in keys:
                self.lam = self.settings["regularization_lambda"]
            else:
                self.lam = default_settings["regularization_lambda"]
        if "optimizer" in keys:
            self.optimizer_key = self.settings["optimizer"]
            self.optimizer = self.OptimizerList[self.optimizer_key]
        if "normalizer" in keys:
            self.norm_key = self.settings["normalizer"]
            self.normalizer = self.NormList[self.norm_key]

    def loss(self, x, y):
        if self.do_regularization:
            return (1 / len(x[0])) * (cp.sum(self.cost(x, y))
                                      + (self.lam/2) * self.regularization.fn(self.weights, self.biases))
        else:
            return (1 / len(x[0])) * cp.sum(self.cost(x, y))

    def make_dict(self):
        self.ActivationList = {
            "TanH": Activation.TanH(),
            "Sigmoid": Activation.Sigmoid(),
            "ReLu": Activation.ReLu(),
            "LeakyReLu": Activation.LReLU(),
            "ParametrizedReLu": Activation.PReLU(self.param),
            "ExponentialLU": Activation.ELU(self.param),
            "SoftPlus": Activation.SoftPlus(),
            "SoftMax": Activation.SoftMax,
            "LinearActivation": Activation.LinearActivation()
        }
        self.OptimizerList = {
            "SGD": self.simple_optimizer,
            "Momentum": self.momentum,
            "RMSprop": self.RMSprop,
            "Adam": self.Adam
        }
        self.RegularizationList = {
            "L2": regularization.L2()
        }
        self.CostList = {
            "MSE": cost.MSE(),
            "MAE": cost.MAE(),
            "CrossEntropy": cost.CrossEntropy()
        }
        self.NormList = {
            "LinearScaling": Normalizer.LinearScaling,
            "LogScaling": Normalizer.logScaling,
            "z_score": Normalizer.Z_score
        }

    def make(self, network_shape, activation):
        for i in network_shape:
            if type(i) == dict:
                if "activation" in i.keys():
                    self.activation.append(self.ActivationList[i["activation"]].fn)
                    self.derivative.append(self.ActivationList[i["activation"]].d)
                    self.Activations.append(i["activation"])
                else:
                    self.activation.append(self.ActivationList[activation].fn)
                    self.derivative.append(self.ActivationList[activation].d)
                    self.Activations.append(activation)
                self.shape.append(i["shape"])
            elif type(i) == int:
                self.activation.append(self.ActivationList[activation].fn)
                self.derivative.append(self.ActivationList[activation].d)
                self.Activations.append(activation)
                self.shape.append(i)
            else:
                raise TypeError("bad network_shape")

    def print_accuracy(self, x, y, do_return=False, do_print=True, find_loss=False, in_percent=False):
        y_hat = self.predict(x)
        output = cp.argmax(y_hat, axis=0)
        target = cp.argmax(y, axis=0)
        correct = cp.equal(output, target)
        correct[correct == False] = 0
        correct[correct == True] = 1
        correct = cp.sum(correct)
        total = len(target)
        per = correct/total
        if find_loss:
            loss = self.loss(y_hat, y)
        if do_print:
            if in_percent:
                if not find_loss:
                    print(f"{correct}/{total} accuracy = {per * 100}%")
                else:
                    print(f"{correct}/{total} accuracy = {per * 100}%, loss = {loss}")
            else:
                if not find_loss:
                    print(f"{correct}/{total} accuracy = {per}")
                else:
                    print(f"{correct}/{total} accuracy = {per}, loss = {loss}")
        if do_return:
            x = [correct, total, per]
            if find_loss:
                x.append(loss)
            return x

    def predict(self, x):
        if self.do_norm:
            output = self.normalizer(x)
        else:
            output = x.copy()
        for i in range(len(self.weights)):
            output = self.activation[i](cp.matmul(self.weights[i], output) + self.biases[i])
        return output

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1,
              train_type="MGD", batch_size=32, shuffle_data=False,
              record_both_errors=True, print_loss=True, return_error=True,
              on_epoch_end=None, on_epoch_start=None, on_batch_end=None, on_batch_start=None):
        if self.do_norm:
            x_t = self.normalizer(x_train)
            y_t = self.normalizer(y_train)
            if x_val is not None:
                x_v = self.normalizer(x_val)
                y_v = self.normalizer(y_val)
        else:
            x_t = x_train.copy()
            y_t = y_train.copy()
            if x_val is not None:
                x_v = x_val.copy()
                y_v = y_val.copy()
        if return_error:
            if not record_both_errors:
                errors = []
            else:
                errors1 = []
                errors2 = []
        for _ in range(epochs):
            if on_epoch_start is not None:
                on_epoch_start(epoch=_, training_data=[x_train, y_train], cross_validation_data=[x_val, y_val],
                               model=self, epochs=epochs, train_type=train_type, batch_size=batch_size,
                               shuffle_data=shuffle_data, record_both_errors=record_both_errors,
                               return_error=return_error, print_loss=print_loss)
            if train_type == "BGD":
                self.BatchGrad(x_t, y_t)
            elif train_type == "MGD":
                self.MiniBatchGrad(x_t, y_t, batch_size, on_batch_end, on_batch_start)
            elif train_type == "SGD":
                self.StochasticGrad(x_t, y_t)
            if x_val is not None and not record_both_errors:
                loss = self.loss(self.predict(x_v), y_v)
                if print_loss:
                    print(f"Cross Validation Loss after Epoch {_+1} = {loss}")
                if return_error:
                    errors.append(loss)
                losses = [loss]
            elif not record_both_errors:
                loss = self.loss(self.predict(x_t), y_t)
                if print_loss:
                    print(f"Train Loss after Epoch {_+1} = {loss}")
                if return_error:
                    errors.append(loss)
                losses = [loss]
            else:
                if x_val is not None:
                    loss1 = self.loss(self.predict(x_v), y_v)
                loss2 = self.loss(self.predict(x_t), y_t)
                if print_loss:
                    if x_val is not None:
                        print(f"Cross Validation Loss after Epoch {_ + 1} = {loss1}")
                    print(f"Train Loss after Epoch {_ + 1} = {loss2}")
                if return_error:
                    if x_val is not None:
                        errors1.append(loss1)
                    errors2.append(loss2)
                if x_val is not None:
                    losses = [loss1, loss2]
                else: losses = [loss2]
            if shuffle_data:
                shuffle = cp.random.permutation(x_t.shape[1])
                x_t = x_t[:, shuffle]
                y_t = y_t[:, shuffle]
            if on_epoch_end is not None:
                on_epoch_end(epoch=_, training_data=[x_train, y_train], cross_validation_data=[x_val, y_val],
                             model=self, loss=losses, epochs=epochs, train_type=train_type, batch_size=batch_size,
                             shuffle_data=shuffle_data, record_both_errors=record_both_errors,
                             return_error=return_error, print_loss=print_loss)
        if return_error:
            if not record_both_errors:
                return errors
            else:
                if x_val is not None:
                    return errors1, errors2
                else:
                    return errors2

    def BatchGrad(self, x, y):
        a = [x]
        z = []
        if self.optimizer_key == "Momentum" or self.optimizer_key == "RMSprop":
            Vdw = [cp.zeros(i.shape) for i in self.weights]
            Vdb = [cp.zeros(i.shape) for i in self.biases]
        elif self.optimizer_key == "Adam":
            Vdw = [cp.zeros(i.shape) for i in self.weights]
            Vdb = [cp.zeros(i.shape) for i in self.biases]
            Sdw = [cp.zeros(i.shape) for i in self.weights]
            Sdb = [cp.zeros(i.shape) for i in self.biases]
        for i in range(len(self.weights)):
            z.append(cp.matmul(self.weights[i], a[i]) + self.biases[i])
            a.append(self.activation[i](z[i]))
        error = self.d_cost(a[len(a)-1], y)
        for i in reversed(range(len(self.weights))):
            if not self.Activations[i] == SoftMax:
                dZ = error*self.derivative[i](z[i])
            else:
                dZ = self.derivative[i](a[i+1], y)
            if not self.do_regularization:
                dW = (1/len(x[0])) * cp.matmul(dZ, a[i].T)
                dB = (1/len(x[0])) * cp.sum(dZ, axis=1, keepdims=True)
            else:
                dW = (1 / len(x[0])) * cp.matmul(dZ, a[i].T) + \
                     self.regularization.fn2(self.weights[i], self.lam, len(x[0]))
                dB = (1 / len(x[0])) * cp.sum(dZ, axis=1, keepdims=True) + \
                     self.regularization.fn2(self.biases[i], self.lam, len(x[0]))
            if self.optimizer_key == "SGD":
                self.optimizer(dW, dB, i)
            elif self.optimizer_key == "Momentum" or self.optimizer_key == "RMSprop":
                self.optimizer(dW, dB, Vdw[i], Vdb[i], i, 1)
            elif self.optimizer_key == "Adam":
                self.optimizer(dW, dB, Vdw[i], Vdb[i], Sdw[i], Sdb[i], i, 1)
            error = cp.matmul(self.weights[i].T, dZ)

    def MiniBatchGrad(self, x_t, y_t, batch_size, on_batch_end, on_batch_start):
        xs = x_t.T
        ys = y_t.T
        if self.optimizer_key == "Momentum" or self.optimizer_key == "RMSprop":
            Vdw = [cp.zeros(i.shape) for i in self.weights]
            Vdb = [cp.zeros(i.shape) for i in self.biases]
        elif self.optimizer_key == "Adam":
            Vdw = [cp.zeros(i.shape) for i in self.weights]
            Vdb = [cp.zeros(i.shape) for i in self.biases]
            Sdw = [cp.zeros(i.shape) for i in self.weights]
            Sdb = [cp.zeros(i.shape) for i in self.biases]
        batches = int(cp.ceil(len(x_t[0]) / batch_size))
        for batch in range(batches):
            if (batch+1)*batch_size < len(x_t[0]):
                x = xs[batch * batch_size:(batch+1) * batch_size].T
                y = ys[batch * batch_size:(batch + 1) * batch_size].T
            else:
                x = xs[batch * batch_size:].T
                y = ys[batch * batch_size:].T
            if on_batch_start is not None:
                on_batch_start(batch=batch, batches=batches, batch_size=batch_size, data=[x, y], model=self)
            a = [x]
            z = []
            for i in range(len(self.weights)):
                z.append(cp.matmul(self.weights[i], a[i]) + self.biases[i])
                a.append(self.activation[i](z[i]))
            error = self.d_cost(a[len(a)-1], y)
            for i in reversed(range(len(self.weights))):
                if not self.Activations[i] == SoftMax:
                    dZ = error * self.derivative[i](z[i])
                else:
                    dZ = self.derivative[i](z[i], y)
                if not self.do_regularization:
                    dW = (1/batch_size) * cp.matmul(dZ, a[i].T)
                    dB = (1 / batch_size) * cp.sum(dZ, axis=1, keepdims=True)
                else:
                    dW = (1/batch_size) * cp.matmul(dZ, a[i].T) + \
                         self.regularization.fn2(self.weights[i], self.lam, len(x_t[0]))
                    dB = (1 / batch_size) * cp.sum(dZ, axis=1, keepdims=True) + \
                         self.regularization.fn2(self.biases[i], self.lam, len(x_t[0]))
                if self.optimizer_key == "SGD":
                    self.optimizer(dW, dB, i)
                elif self.optimizer_key == "Momentum" or self.optimizer_key == "RMSprop":
                    self.optimizer(dW, dB, Vdw[i], Vdb[i], i, batch)
                elif self.optimizer_key == "Adam":
                    self.optimizer(dW, dB, Vdw[i], Vdb[i], Sdw[i], Sdb[i], i, batch)
                error = cp.matmul(self.weights[i].T, dZ)
            if on_batch_end is not None:
                on_batch_end(batch=batch, batches=batches, batch_size=batch_size, data=[x, y], model=self)

    def StochasticGrad(self, x_t, y_t):
        x = x_t.T
        y = y_t.T
        if self.optimizer_key == "Momentum" or self.optimizer_key == "RMSprop":
            Vdw = [cp.zeros(i.shape) for i in self.weights]
            Vdb = [cp.zeros(i.shape) for i in self.biases]
        elif self.optimizer_key == "Adam":
            Vdw = [cp.zeros(i.shape) for i in self.weights]
            Vdb = [cp.zeros(i.shape) for i in self.biases]
            Sdw = [cp.zeros(i.shape) for i in self.weights]
            Sdb = [cp.zeros(i.shape) for i in self.biases]
        for j in range(len(x_t[0])):
            a = [cp.array([x[0:][j]]).T]
            z = []
            for i in range(len(self.weights)):
                z.append(cp.matmul(self.weights[i], a[i]) + self.biases[i])
                a.append(self.activation[i](z[i]))
            error = self.d_cost(a[len(a)-1], cp.array([y[0:][j]]).T)
            for i in reversed(range(len(self.weights))):
                if not self.Activations[i] == SoftMax:
                    dz = error * self.derivative[i](z[i])
                else:
                    dz = self.derivative[i](z[i], y[i:i+1].T)
                if not self.do_regularization:
                    dw = cp.matmul(dz, a[i].T)
                    db = dz
                else:
                    dw = cp.matmul(dz, a[i].T) + self.regularization.fn2(self.weights[i], self.lam, len(x_t[0]))
                    db = dz + self.regularization.fn2(self.biases[i], self.lam, 1)
                if self.optimizer_key == "SGD":
                    self.optimizer(dw, db, i)
                elif self.optimizer_key == "Momentum" or self.optimizer_key == "RMSprop":
                    self.optimizer(dw, db, Vdw[i], Vdb[i], i, j)
                elif self.optimizer_key == "Adam":
                    self.optimizer(dw, db, Vdw[i], Vdb[i], Sdw[i], Sdb[i], i, j)
                error = cp.matmul(self.weights[i].T, dz)

    def print_network(self):

        # Functions

        print(self.ActivationList)  # Activation functions list
        print(self.OptimizerList)  # Optimizer functions list
        print(self.RegularizationList)  # Regularization functions list
        print(self.CostList)  # Cost function list
        print(self.NormList)  # Normalizing function list
        print(self.optimizer)  # optimizer function
        print(self.activation)  # activation function list
        print(self.derivative)  # derivative function list
        print(self.cost)  # cost function
        print(self.d_cost)  # derivative of cost function
        print(self.regularization)  # regularization function
        print(self.normalizer)  # data normalizing function

        # Variables

        print(self.shape)  # shape of the neural network
        print(self.weights_shape)  # shape of the weights in neural network
        print(self.weights)  # weights of the neural network
        print(self.biases)  # biases of neural network
        print(self.learning_rate)  # learning rate
        print(self.Activations)  # activations name list
        print(self.param)  # activations parameter
        print(self.optimizer_key)  # optimizer function name
        print(self.cost_key) # cost function name
        print(self.beta)  # optimizer parameter
        print(self.gamma)  # optimizer parameter
        print(self.epsilon)  # optimizer parameter
        print(self.do_regularization)  # do regularization
        print(self.regularization_key)  # regularization function name
        print(self.norm_key)  # normalizing function name
        print(self.lam)  # regularization parameter

    def save(self, file_name):
        weights = [i.tolist() for i in self.weights]
        biases = [i.tolist() for i in self.biases]
        temp = {
            "shape": self.shape,
            "weights_shape": self.weights_shape,
            "learning_rate": self.learning_rate,
            "Activations": self.Activations,
            "Activation param": self.param,
            "Optimizer": self.optimizer_key,
            "Cost": self.cost_key,
            "beta": self.beta,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "do_regularization": self.do_regularization,
            "regularization_key": self.regularization_key,
            "do_norm": self.do_norm,
            "normalizer": self.norm_key,
            "weights": weights,
            "biases": biases
        }
        temp = json.dumps(temp, indent=3)
        with open(file_name, "w") as file:
            file.write(temp)


def load(file_name):
    with open(file_name, "r") as file:
        data = json.loads(file.read())
        model = NeuralNetwork()
        model.shape = data["shape"]
        model.weights_shape = data["weights_shape"]
        model.learning_rate = data["learning_rate"]
        model.Activations = data["Activations"]
        model.param = data["Activation param"]
        model.optimizer_key = data["Optimizer"]
        model.cost_key = data["Cost"]
        model.beta = data["beta"]
        model.gamma = data["gamma"]
        model.epsilon = data["epsilon"]
        model.do_regularization = data["do_regularization"]
        model.regularization_key = data["regularization_key"]
        model.do_norm = data["do_norm"]
        model.norm_key = data["normalizer"]
        weights = [cp.array(i) for i in data["weights"]]
        model.weights = weights
        biases = [cp.array(i) for i in data["biases"]]
        model.biases = biases
        model.activation = [model.ActivationList[i].fn for i in model.Activations]
        model.derivative = [model.ActivationList[i].d for i in model.Activations]
        model.optimizer = model.OptimizerList[model.optimizer_key]
        model.cost = model.CostList[model.cost_key].cost
        model.d_cost = model.CostList[model.cost_key].d_cost
        model.normalizer = model.NormList[model.norm_key]
        if model.do_regularization:
            model.regularization = model.RegularizationList[model.regularization_key]
    return model
