from .gene import Gene
import torch.optim as optim
import numpy as np


class Optimizer(Gene):

    def __init__(self):
        super().__init__()
        self.learning_rate = None
        self.optimizer = None
        self.optimizer_conf = None

    def create_phenotype(self, phenotype):
        self.optimizer_conf = phenotype.optimizer_conf
        if self.learning_rate is None:
            self.learning_rate = phenotype.optimizer_conf.learning_rate
        if phenotype.optimizer_conf.type == "Adam":
            self.optimizer = optim.Adam(phenotype.parameters(), lr=self.learning_rate, betas=(0, 0.99), weight_decay=phenotype.optimizer_conf.weight_decay)
        elif phenotype.optimizer_conf.type == "SGD":
            self.optimizer = optim.SGD(phenotype.parameters(), nesterov=True, lr=self.learning_rate, momentum=0.95, weight_decay=phenotype.optimizer_conf.weight_decay)
        elif phenotype.optimizer_conf.type == "RMSprop":
            self.optimizer = optim.RMSprop(phenotype.parameters(), lr=self.learning_rate, weight_decay=phenotype.optimizer_conf.weight_decay)
        elif phenotype.optimizer_conf.type == "Adadelta":
            self.optimizer = optim.Adadelta(phenotype.parameters(), weight_decay=phenotype.optimizer_conf.weight_decay)
        else:
            clazz = getattr(optim, phenotype.optimizer_conf.type)
            self.optimizer = clazz(phenotype.parameters(), lr=self.learning_rate, weight_decay=phenotype.optimizer_conf.weight_decay)
        return self.optimizer

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'learning_rate=' + str(self.learning_rate) + ')'

    # def apply_mutation(self):
    #     self.learning_rate += np.random.normal(0.0, self.learning_rate/10)
    #     self.learning_rate = min(max(self.optimizer_conf.learning_rate/10, self.learning_rate), 2*self.optimizer_conf.learning_rate)
    #     print("MUTATE optimizer", self.learning_rate)
