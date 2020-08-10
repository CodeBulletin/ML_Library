class optimizers:
    def simple_optimizer(self, dW, dB, i):
        self.weights[i] -= self.learning_rate*dW
        self.biases[i] -= self.learning_rate*dB

    def momentum(self, dW, dB, Vdw, Vdb, i, j):
        Vdw = (self.beta*Vdw + (1-self.beta)*dW)/(1+self.beta**j)
        Vdb = (self.beta*Vdb + (1-self.beta)*dB)/(1+self.beta**j)
        self.weights[i] -= self.learning_rate * Vdw
        self.biases[i] -= self.learning_rate * Vdb

    def RMSprop(self, dW, dB, Sdw, Sdb, i, j):
        Sdw = (self.beta*Sdw + (1-self.beta)*(dW**2))/(1+self.beta**j)
        Sdb = (self.beta*Sdb + (1-self.beta)*(dB**2))/(1+self.beta**j)
        self.weights[i] -= self.learning_rate * dW/(Sdw**0.5 + self.epsilon)
        self.biases[i] -= self.learning_rate * dB/(Sdb**0.5 + self.epsilon)

    def Adam(self, dW, dB, Vdw, Vdb, Sdw, Sdb, i, j):
        Vdw = (self.beta * Vdw + (1 - self.beta) * dW)/(1+self.beta**j)
        Vdb = (self.beta * Vdb + (1 - self.beta) * dB)/(1+self.beta**j)
        Sdw = (self.gamma * Sdw + (1 - self.gamma) * (dW ** 2))/(1+self.gamma**j)
        Sdb = (self.gamma * Sdb + (1 - self.gamma) * (dB ** 2))/(1+self.gamma**j)
        self.weights[i] -= self.learning_rate * Vdw/(Sdw**0.5 + self.epsilon)
        self.biases[i] -= self.learning_rate * Vdb / (Sdb ** 0.5 + self.epsilon)
