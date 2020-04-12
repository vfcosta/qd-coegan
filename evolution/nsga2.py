# based on https://github.com/haris989/NSGA-II/blob/master/NSGA%20II.py
# also based on https://github.com/ChengHust/NSGA-II
import random
import sys
import numpy as np
from evolution.base_evolution import BaseEvolution
from evolution.population import Population
import logging
from munch import Munch
from evolution.config import config

logger = logging.getLogger(__name__)


class NSGA2Base:

    def __init__(self, use_crowding_distance=True):
        self.use_crowding_distance = use_crowding_distance
        self.generation = 0
        self.generate_child = None  # Function used to create a child based on parents

    def sort_by_values(self, list1, values):
        combined = sorted(zip(list1, np.array(values)[list1]), key=lambda x: x[1])
        return [c[0] for c in combined]

    def dominates(self, all_values, i, j):
        if i == j:
            return False
        any_greater = False
        for values in all_values:
            if values[i] < values[j]:
                return False
            any_greater |= values[i] > values[j]
        return any_greater

    def fast_non_dominated_sort(self, all_values):
        size = len(all_values[0])
        S = [[]] * size
        front = [[]]
        n = [0] * size
        for p in range(size):
            S[p] = []
            n[p] = 0
            for q in range(size):
                if self.dominates(all_values, p, q):
                    if q not in S[p]:
                        S[p].append(q)
                elif self.dominates(all_values, q, p):
                    n[p] = n[p] + 1
            if n[p] == 0:
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while front[i]:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        if q not in Q:
                            Q.append(q)
            yield i, front[i]
            i += 1
            front.append(Q)

    def crowding_distance(self, all_values, front):
        distance = [0] * len(front)
        distance[0], distance[-1] = sys.maxsize, sys.maxsize
        for values in all_values:
            sorted_list = self.sort_by_values(front, values)
            for k in range(1, len(front) - 1):
                distance[k] += (values[sorted_list[k + 1]] - values[sorted_list[k - 1]]) / np.ptp(values)
        return distance

    def tournament(self, population, k=2):
        n = len(population)
        parents = []
        for i in range(n):
            a = np.random.randint(n)
            for j in range(k):
                b = np.random.randint(n)
                if self.crowded_operator(population[b], population[a]):
                    a = b
            parents.append(population[a])
        return parents

    def crowded_operator(self, s1, s2):
        return s1.rank < s2.rank or (s1.rank == s2.rank and s1.crowding_distance > s2.crowding_distance)

    def generate_children(self, population):
        parents, mate = self.tournament(population), self.tournament(population)
        return [self.generate_child([p, m]) for p, m in zip(parents, mate)]

    def generate_next_population(self, population, children):
        pop_size = len(population)
        all_population = population + children
        all_values = [[p.objectives[i] for p in all_population] for i in range(len(all_population[0].objectives))]
        non_dominated = self.fast_non_dominated_sort(all_values)
        new_population = []

        for rank, front in non_dominated:
            remaining = pop_size - len(new_population)
            logger.info(f"getting individuals from front {rank} ({len(front)}/{remaining})")
            if self.generation % 10 == 0 and rank == 0:
                logger.info(f"best front for generation {self.generation}:")
                logger.info(" ".join(str(all_population[v]) for v in front))

            # sort only if the front is bigger than the elements needed to fill the population
            if self.use_crowding_distance:
                logger.info(f"apply crowding distance at generation {self.generation} ({len(front)} > {remaining})")
                distance = self.crowding_distance(all_values, front)
                sorted_indexes = list(reversed(self.sort_by_values(range(len(front)), distance)))
                front = [front[i] for i in sorted_indexes]
                distance = [distance[i] for i in sorted_indexes]
                # print("front", self.generation, front, sorted_indexes, distance)
            else:
                distance = [sys.maxsize] * len(front)

            selected_indexes = front[:min(len(front), remaining)]
            for i, index in enumerate(selected_indexes):
                all_population[index].rank = rank
                all_population[index].crowding_distance = distance[i]
            new_population += selected_indexes
            if len(new_population) == pop_size:
                break
        self.generation += 1
        return [all_population[i] for i in new_population]

    def start_demo(self, plot_results=True, pop_size=20, max_gen=500, bounds=[-55, 55]):
        functions = [lambda x: -x.value ** 2, lambda x: -(x.value - 2) ** 2]
        mutation = lambda x, prob=1: bounds[0] + np.ptp(bounds) * random.random() if random.random() < prob else x

        def generate_child(parents):
            child = Munch.fromDict({"value": mutation(np.mean([p.value for p in parents])), "rank": None})
            child.objectives = [f(child) for f in functions]
            return child

        self.generate_child = generate_child
        population = [generate_child([]) for _ in range(pop_size)]
        children = []
        for _ in range(max_gen):
            population = self.generate_next_population(population, children)
            children = self.generate_children(population)
        if plot_results:
            import matplotlib.pyplot as plt
            plt.xlabel('Function 1')
            plt.ylabel('Function 2')
            plt.scatter(*[[np.array([f(i) for i in population]) * -1] for f in functions[:2]])
            plt.show()


# TODO keep the best unchanged even when training
# is the problem the disciminator's metric?
class NSGA2(BaseEvolution):

    def __init__(self, evaluator):
        super().__init__(evaluator)
        self.nsga2_base = NSGA2Base(use_crowding_distance=False)
        self.nsga2_base.generate_child = self.generate_child
        self.nsga2_base.crowded_operator = self.crowded_operator
        self.archive_d, self.archive_g = [], []
        self.min_fitness_d, self.min_fitness_g = None, None

    def crowded_operator(self, s1, s2, fitness_percent=2.0):
        # fitness_constraint = self.min_fitness_d if isinstance(s1, Discriminator) else self.min_fitness_g
        fitness_constraint = min(s1.fitness(), s2.fitness())
        fitness_constraint *= fitness_percent if fitness_constraint >= 0 else 1/fitness_percent
        feasible_s1, feasible_s2 = s1.fitness() <= fitness_constraint, s2.fitness() <= fitness_constraint
        logger.info(f"fitness: {s1.fitness()}, {s2.fitness()}, fitness_constraint: {fitness_constraint}")
        if feasible_s1 and not feasible_s2:
            logger.info(f"s2 not feasible")
            return True
        if not feasible_s1 and not feasible_s2:
            logger.info(f"both not feasible")
            return s1.fitness() < s2.fitness()
        return s1.rank < s2.rank or (s1.rank == s2.rank and s1.fitness() < s2.fitness())

    def next_population(self, generators_population, discriminators_population):
        logger.debug(f"[generation {self.generation}] evaluate population")
        if self.generation == 0:
            g_children, d_children = [], []
        else:
            g_children = self.nsga2_base.generate_children(generators_population.phenotypes())
            d_children = self.nsga2_base.generate_children(discriminators_population.phenotypes())

        # for p in g_children + generators_population.phenotypes():
        #     self.evaluator.evaluate_population([p], [discriminators_population.best().clone()])
        # for p in d_children + discriminators_population.phenotypes():
        #     self.evaluator.evaluate_population([generators_population.best().clone()], [p], calc_fid=False)
        self.evaluator.evaluate_population(g_children, discriminators_population.phenotypes())
        self.evaluator.evaluate_population(generators_population.phenotypes(), d_children)

        self.evaluate_population(g_children + generators_population.phenotypes(), d_children + discriminators_population.phenotypes())
        self.add_archive(g_children, self.archive_g)
        self.add_archive(d_children, self.archive_d)
        generators = self.nsga2_base.generate_next_population(generators_population.phenotypes(), g_children)
        discriminators = self.nsga2_base.generate_next_population(discriminators_population.phenotypes(), d_children)
        self.min_fitness_d = min(d.fitness() for d in discriminators)
        self.min_fitness_g = min(g.fitness() for g in generators)
        gen_pop = Population(generators, stats={"archive_len": len(self.archive_g)})
        dis_pop = Population(discriminators, stats={"archive_len": len(self.archive_d)})
        return gen_pop, dis_pop

    def add_archive(self, individuals, archive, max_size=500, prob=config.evolution.nslc.archive_prob):
        for p in individuals:
            if np.random.rand() < prob:
                archive.append(Munch({"genome": p.genome.clean_copy(), "fitness": p.fitness()}))
            if len(archive) > max_size:
                archive.pop(0)

    def evaluate_population(self, generators, discriminators):
        if self.evaluator.initial:
            # g, d = generators[0], discriminators[0]
            # for p in generators:
            #     self.evaluator.evaluate_population([p], [d.clone()])
            # for p in discriminators:
            #     self.evaluator.evaluate_population([g.clone()], [p], calc_fid=False)
            self.evaluator.evaluate_population(generators, discriminators)
            self.evaluator.initial = False

        # functions = [lambda x: -x.fitness(), lambda x: -x.genome.age]
        neighbors_size = config.evolution.nslc.neighbors_size
        for population, archive in [(generators, self.archive_g), (discriminators, self.archive_d)]:
            print("archive", len(archive))
            archive = archive + [Munch({"genome": p.genome, "fitness": p.fitness()}) for p in population]
            for i, p in enumerate(population):
                distances = {n: p.genome.distance(archive[n].genome) for n in range(len(archive)) if archive[n].genome != p.genome}
                distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
                neighbors = list(distances.keys()) if neighbors_size == "all" else list(distances.keys())[:neighbors_size]
                fitness = np.array([archive[n].fitness for n in neighbors])
                print(i, neighbors, list(distances.values()), fitness, p.fitness())
                local_competition = len(np.where(fitness >= p.fitness())[0])
                novelty = np.mean([distances[n] for n in neighbors])
                p.objectives = [local_competition, novelty]
                print("objectives", p.objectives)


if __name__ == "__main__":
    nsga2 = NSGA2Base(use_crowding_distance=True)
    nsga2.start_demo()
