from .config import config
import numpy as np
import util.tools as tools
from .population import Population
from .base_evolution import BaseEvolution
import logging

logger = logging.getLogger(__name__)


class NEAT(BaseEvolution):

    def select_spatial(self, population, k=2):
        selected = []
        all_individuals = population.phenotypes()
        rows = 3
        cols = len(all_individuals) // rows
        for center in range(len(all_individuals)):
            neighbor_indexes = tools.get_neighbors(center, rows, cols)
            indexes = np.random.choice(neighbor_indexes, k, replace=False)
            individuals = [(all_individuals[ix].fitness(), ix) for ix in indexes]
            individuals.sort()
            print("individuals", individuals, len(individuals))
            selected.append((all_individuals[individuals[0][-1]], all_individuals[individuals[0][-1]]))

            # neighbors = []
            # min_fitness = np.mean([all_individuals[i].fitness() for i in neighbor_indexes])
            # min_fitness = min([all_individuals[i].fitness() for i in neighbor_indexes]) * 0.95
            # for n in neighbor_indexes:
            #     distance = np.mean(
            #         [all_individuals[n].genome.distance(all_individuals[n2].genome) for n2 in
            #          neighbor_indexes if n != n2])
            #     # neighbors.append((np.sign(all_individuals[n].fitness() - min_fitness), -distance, n))
            #     neighbors.append((all_individuals[n].fitness(), -distance, n))
            # neighbors.sort()
            #
            # # neighbors = sorted([(all_individuals[i].fitness(), i) for i in tools.get_neighbors(center, rows, cols)])
            # print("neighbors", neighbors, len(neighbors))
            # selected.append((all_individuals[neighbors[0][-1]].clone(), all_individuals[neighbors[0][-1]].clone()))
        return [selected]  # wrap in an array to represent a single specie

    def select(self, population, discard_percent=0, k=config.evolution.tournament_size):
        """Select individuals based on fitness sharing"""

        if config.evolution.evaluation.type == "spatial":
            return self.select_spatial(population)

        ### TOURNAMENT TEST
        # population_size = len(population.phenotypes())
        # phenotypes = population.phenotypes()
        # selected = []
        # for i in range(population_size):
        #     p = np.random.choice(phenotypes, 3, replace=False).tolist()
        #     p.sort(key=lambda x: x.fitness())
        #     selected.append([p[0], p[0]])
        # return [selected]
        ###

        population_size = len(population.phenotypes())
        species_selected = []
        species_list = population.species_list
        average_species_fitness_list = []
        for species in species_list[:]:
            species.remove_invalid()  # discard invalid individuals
            if len(species) > 0:
                average_species_fitness_list.append(species.average_fitness())
            else:
                species_list.remove(species)
        total_fitness = np.sum(average_species_fitness_list)

        # initialize raw sizes with equal proportion
        raw_sizes = [population_size / len(species_list)] * len(species_list)
        if total_fitness != 0:
            # calculate proportional sizes when total fitness is not zero
            raw_sizes = [average_species_fitness / total_fitness * population_size
                         for average_species_fitness in average_species_fitness_list]

        sizes = tools.round_array(raw_sizes, max_sum=population_size, invert=True)

        for species_obj, size in zip(species_list, sizes):
            size = int(size)
            # discard the lowest-performing individuals
            species = species_obj.best_percent(1 - discard_percent)

            # tournament selection inside species
            selected = []

            # ensure that the best was selected
            if config.evolution.speciation.keep_best and size > 0:
                selected.append([species[0]])

            orig_species = list(species)
            for i in range(size - len(selected)):
                parents = []
                for l in range(2):
                    winner = None
                    for j in range(k):
                        random_index = np.random.randint(0, len(species))
                        if winner is None or species[random_index].fitness() < winner.fitness():
                            winner = species[random_index]
                        del species[random_index]  # remove element to emulate draw without replacement
                        if len(species) == 0:  # restore original list when there is no more individuals to draw
                            species = list(orig_species)
                    parents.append(winner)
                    if config.evolution.crossover_rate == 0:
                        # do not draw another individual from the population if there is no probability of crossover
                        break
                selected.append(parents)

            species_selected.append(selected)
        return species_selected

    def generate_children(self, species_list):
        # generate child (only mutation for now)
        children = []
        for species in species_list:
            for i, parents in enumerate(species):
                children.append(self.generate_child(parents, is_best=i == 0))
        return children

    def replace_population(self, population, children):
        elite = population.best_percent(config.evolution.elitism)
        children = sorted(children, key=lambda x: x.fitness())
        new_population = Population(
            elite + children[:len(children) - len(elite)], desired_species=config.evolution.speciation.size,
            speciation_threshold=population.speciation_threshold)
        return new_population

    def next_population(self, generators_population, discriminators_population):
        logger.info("next population")
        # select parents for reproduction
        g_parents = self.select(generators_population)
        d_parents = self.select(discriminators_population)
        # apply variation operators (only mutation for now)
        g_children = self.generate_children(g_parents)

        # limit the number of layers in D's to the max layers among G's
        if config.evolution.control_number_layers:
            self.limit_number_layers(d_parents, g_children)

        d_children = self.generate_children(d_parents)
        # evaluate the children population and the best individuals (when elitism is being used)
        logger.debug(f"[generation {self.generation}] evaluate population")
        self.evaluator.evaluate_population(g_children, d_children)
        # generate a new population based on the fitness of the children and elite individuals
        return self.replace_population(generators_population, g_children), \
            self.replace_population(discriminators_population, d_children)
