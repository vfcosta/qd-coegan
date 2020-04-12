import util.tools as tools
from .config import config
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self, train_loader, validation_loader):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.train_loader_iter = iter(self.train_loader)
        self.best_discriminators = []
        self.best_generators = []
        self.initial = True

    def train_evaluate(self, G, D, norm_g=1, norm_d=1):
        if config.evolution.evaluation.reset_optimizer:
            D.reset_optimizer_state()
            G.reset_optimizer_state()

        if G.invalid or D.invalid:  # do not evaluate if G or D are invalid
            logger.warning("invalid D or G")
            return
        torch.cuda.empty_cache()
        n, ng = 0, 0
        G.error = G.error or 0
        D.error = D.error or 0
        g_error = G.error
        d_error = D.error
        d_fitness_value, g_fitness_value = D.fitness_value, G.fitness_value
        G, D = tools.cuda(G), tools.cuda(D)  # load everything on gpu (cuda)
        G.train()
        D.train()
        G.win_rate, D.win_rate = 0, 0
        while n < config.gan.batches_limit:
            for images, _ in self.train_loader_iter:
                if n + 1 > config.gan.batches_limit:
                    break
                n += 1
                images = tools.cuda(images)
                D.do_train(G, images)
                if n % config.gan.critic_iterations == 0:
                    ng += 1
                    G.do_train(D, images)
            else:
                self.train_loader_iter = iter(self.train_loader)
        D.error = d_error + (D.error - d_error)/(n*norm_d)
        new_d_fitness = (D.fitness_value - d_fitness_value) / n
        new_g_fitness = (G.fitness_value - g_fitness_value) / ng
        D.fitness_value = d_fitness_value + new_d_fitness / norm_d
        G.fitness_value = g_fitness_value + new_g_fitness / norm_g
        G.error = g_error + (G.error - g_error)/(ng*norm_g)

        D.win_rate /= n
        G.win_rate = 1 - D.win_rate
        D.calc_skill_rating(G)
        G.calc_skill_rating(D)
        # print("train GLICKO G:", G.skill_rating, G.win_rate, ", D:", D.skill_rating, D.win_rate)

        G.cpu(), D.cpu()  # move variables back from gpu to cpu
        torch.cuda.empty_cache()

    def evaluate_population(self, generators, discriminators, evaluation_type=config.evolution.evaluation.type, calc_fid=True):
        """Evaluate the population using all-vs-all pairing strategy"""

        for i in range(config.evolution.evaluation.iterations):
            if evaluation_type == "random":
                for D in discriminators:
                    for g in np.random.choice(generators, 2, replace=False):
                        self.train_evaluate(g, D, norm_d=2, norm_g=len(discriminators))
                for G in generators:
                    for d in np.random.choice(discriminators, 2, replace=False):
                        self.train_evaluate(G, d, norm_d=len(generators), norm_g=2)
            elif evaluation_type == "spatial":
                rows = 3
                cols = len(discriminators)//rows
                pairs = []
                for center in range(len(discriminators)):
                    pairs.append([(center, n) for n in tools.get_neighbors(center, rows, cols)])
                # reorder pairs to avoid sequential training
                pairs = np.transpose(np.array(pairs), (1, 0, 2)).reshape(-1, 2)
                norm = pairs.shape[0] // len(discriminators)
                for g, d in pairs:
                    self.train_evaluate(generators[g], discriminators[d], norm_d=norm, norm_g=norm)
            elif evaluation_type == "spatial2":
                rows = 3
                cols = len(discriminators)//rows
                for center in range(len(discriminators)):
                    neighbors = tools.get_neighbors(center, rows, cols)
                    norm = len(neighbors)
                    for n in neighbors:
                        self.train_evaluate(generators[center], discriminators[n].clone(), norm_d=norm, norm_g=norm)
                        self.train_evaluate(generators[n].clone(), discriminators[center], norm_d=norm, norm_g=norm)

            elif evaluation_type == "all-vs-all":
                # train all-vs-all in a non-sequential order
                pairs = tools.permutations(generators, discriminators)
                for g, d in pairs:
                    self.train_evaluate(generators[g], discriminators[d], norm_d=len(generators), norm_g=len(discriminators))
            elif evaluation_type in ["all-vs-best", "all-vs-species-best", "all-vs-kbest"]:
                if config.evolution.evaluation.initialize_all and self.initial:
                    self.initial = False
                    # as there are no way to determine the best G and D, we rely on all-vs-all for the first evaluation
                    return self.evaluate_population(generators, discriminators, evaluation_type="all-vs-all")

                pairs = tools.permutations(discriminators, self.best_generators)
                for d, g in pairs:
                    self.train_evaluate(self.best_generators[g], discriminators[d], norm_d=len(self.best_generators), norm_g=len(discriminators))
                pairs = tools.permutations(generators, self.best_discriminators)
                for g, d in pairs:
                    self.train_evaluate(generators[g], self.best_discriminators[d], norm_d=len(generators), norm_g=len(self.best_discriminators))

        # reset FID
        for G in generators:
            G.fid_score = None

        if calc_fid and (config.evolution.fitness.generator == "FID" or config.stats.calc_fid_score):
            for G in generators:
                G.calc_fid()
        # # update the skill rating for the next generation
        for p in discriminators + generators + self.best_discriminators + self.best_generators:
            p.finish_calc_skill_rating()

    def evaluate_all_validation(self, generators, discriminators):
        # evaluate in validation
        logger.info(f"best G: {len(self.best_generators)}, best D: {len(self.best_discriminators)}")
        for D in discriminators:
            for G in self.best_generators + generators:
                with torch.no_grad():
                    self.evaluate_validation(G, D)
        for G in generators:
            for D in self.best_discriminators:
                with torch.no_grad():
                    self.evaluate_validation(G, D)

        # # update the skill rating for the next generation
        for p in discriminators + generators + self.best_discriminators + self.best_generators:
            p.finish_calc_skill_rating()

    def update_bests(self, generators_population, discriminators_population):
        # store best of generation in coevolution memory
        self.best_discriminators = self.get_bests(discriminators_population, self.best_discriminators)
        self.best_generators = self.get_bests(generators_population, self.best_generators)

    def evaluate_validation(self, G, D, eval_generator=True, eval_discriminator=True):
        if G.invalid or D.invalid:  # do not evaluate if G or D are invalid
            logger.warning("invalid D or G")
            return
        torch.cuda.empty_cache()
        G, D = tools.cuda(G), tools.cuda(D)
        G.eval(), D.eval()
        G.win_rate, D.win_rate = 0, 0
        n = 0
        for images, _ in self.validation_loader:
            if n + 1 > config.evolution.fitness.evaluation_batches:
                break
            n += 1
            images = tools.cuda(images)
            G.win_rate, D.win_rate = 0, 0
            D.do_eval(G, images)
        # D.win_rate /= n
            if eval_discriminator:
                D.calc_skill_rating(G)
            if eval_generator:
                G.win_rate = 1 - D.win_rate
                G.calc_skill_rating(D)
        print("eval GLICKO G:", G.skill_rating, G.win_rate, ", D:", D.skill_rating, D.win_rate)
        G, D = G.cpu(), D.cpu()  # move variables back from gpu to cpu
        torch.cuda.empty_cache()

    def get_bests(self, population, previous_best=[]):
        if config.evolution.evaluation.type == "all-vs-species-best":
            return [species.best() for species in population.species_list]
        elif config.evolution.evaluation.type == "all-vs-best":
            return (population.bests(1) + previous_best)[:config.evolution.evaluation.best_size]
        elif config.evolution.evaluation.type == "all-vs-kbest":
            return population.bests(config.evolution.evaluation.best_size)
        return (population.bests(1) + previous_best)[:config.evolution.evaluation.best_size]
