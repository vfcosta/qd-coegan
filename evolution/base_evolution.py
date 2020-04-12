from .population import Population
from .config import config
from abc import abstractmethod


class BaseEvolution:

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.generation = 0

    def compute_generation(self, generators_population, discriminators_population):
        # store best of generation in coevolution memory
        self.evaluator.update_bests(generators_population, discriminators_population)
        generators_population, discriminators_population = self.next_population(generators_population, discriminators_population)
        self.generation += 1
        return generators_population, discriminators_population

    @abstractmethod
    def next_population(self, generators_population, discriminators_population):
        raise Exception("NotImplementedException")

    def generate_child(self, parents, is_best=False):
        mate = parents[1] if len(parents) > 1 else None
        child = parents[0].breed(mate=mate, skip_mutation=config.evolution.speciation.keep_best and is_best,
                                 freeze=config.evolution.freeze_best and is_best)
        child.genome.generation = self.generation
        return child

    def limit_number_layers(self, d_parents, g_children):
        max_layers_g = max([len(gc.genome.genes) for gc in g_children])
        for s in d_parents:
            for dp in s:
                dp[0].genome.max_layers = max_layers_g

    def intialize_population(self, size, class_, **params):
        individuals = []
        for i in range(size):
            ind = class_(**params)
            ind.setup()
            individuals.append(ind)
        return Population(individuals, desired_species=config.evolution.speciation.size)

    def evaluate_population(self, generators, discriminators):
        self.evaluator.evaluate_population(generators, discriminators)
