import torch
import torch.nn as nn
from .genes import Linear, Layer, Deconv2d
from .layers.reshape import Reshape
import numpy as np
import copy
import traceback
from .config import config
import logging
import json
from metrics.glicko2 import glicko2

logger = logging.getLogger(__name__)


class Phenotype(nn.Module):

    def forward(self, x):
        try:
            out = self.model(x)
            return out
        except Exception as err:
            logger.error(err)
            traceback.print_exc()
            self.optimizer = None
            self.invalid = True

    def __init__(self, output_size, genome=None, input_shape=None, optimizer_conf={}):
        super().__init__()
        self.genome = genome
        self.optimizer = None
        self.optimizer_conf = optimizer_conf
        self.model = None
        self.output_size = output_size
        self.error = None
        self.fitness_value = 0
        self.win_rate = 0
        self.invalid = False
        self.input_shape = input_shape
        self.trained_samples = 0
        self.random_fitness = None
        self.glicko = glicko2.Glicko2(mu=1500, phi=350, sigma=config.evolution.fitness.skill_rating.sigma, tau=config.evolution.fitness.skill_rating.tau)
        self.skill_rating = self.glicko.create_rating()
        self.last_error = None
        self.skill_rating_games = []
        self.rank = None
        self.crowding_distance = None
        self.objectives = []
        if config.gan.type in ["rsgan", "rasgan"]:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCELoss()

    def breed(self, mate=None, skip_mutation=False, freeze=False):
        mate_genome = mate.genome if mate else None
        genome = self.genome.breed(skip_mutation=skip_mutation, mate=mate_genome, freeze=freeze)
        p = self.__class__(output_size=self.output_size, genome=genome, input_shape=self.input_shape,
                           optimizer_conf=self.optimizer_conf)
        try:
            p.setup()
            self.copy_to(p)
            if mate:
                mate.copy_to(p)
            p.model.zero_grad()
        except Exception as err:
            logger.error(err)
            traceback.print_exc()
            logger.debug(genome)
            p.optimizer = None
            p.invalid = True
            p.error = 100
            if not skip_mutation or mate is not None:
                logger.debug("fallback to parent copy")
                return self.breed(mate=None, skip_mutation=True)
        return p

    def setup(self):
        # create some input data
        with torch.no_grad():
            x = torch.randn([1] + list(self.input_shape[1:]))
        self.create_model(x)

    def clone(self):
        self.genome.optimizer_gene.optimizer = None
        genome = copy.deepcopy(self.genome)
        p = self.__class__(output_size=self.output_size, genome=genome, input_shape=self.input_shape)
        try:
            p.setup()
            self.copy_to(p)
            p.model.zero_grad()
        except Exception as err:
            logger.error(err)
        return p

    def copy_to(self, target):
        """
        Copy the phenotype parameters to the target.
        This copy will keep the parameters that match in size from the optimizer.
        """
        target.last_error = self.last_error
        target.trained_samples = self.trained_samples
        # if not target.genome.mutated:
        target.skill_rating = glicko2.Rating(self.skill_rating.mu, self.skill_rating.phi, self.skill_rating.sigma)
        # else:
            # target.skill_rating = self.glicko.create_rating()
            # target.skill_rating.mu = min(self.skill_rating.mu, target.skill_rating.mu)
        if not self.optimizer_conf.get("copy_optimizer_state"):
            return

        old_state_dict = self.optimizer.state_dict()
        if len(old_state_dict['state']) == 0:
            return  # there is no state to copy

        # this causes a memory leak with Adam optimizer
        for gene in target.genome.genes:
            old_gene = self.genome.get_gene_by_uuid(gene.uuid)
            if old_gene is None:
                continue
            for (_, param), (_, old_param) in zip(gene.named_parameters(), old_gene.named_parameters() or {}):
                if id(old_param) not in old_state_dict['state']:
                    continue
                old_state = old_state_dict['state'][id(old_param)]
                if ('momentum_buffer' in old_state and old_state['momentum_buffer'].size() == param.data.size()) or \
                        ('exp_avg' in old_state and old_state['exp_avg'].size() == param.data.size()) or \
                        ('square_avg' in old_state and old_state['square_avg'].size() == param.data.size()):
                    target.optimizer.state[param] = copy.deepcopy(old_state)

    def do_train(self, phenotype, images):
        # if self.genome.freeze:
        #     return self.last_error
        if phenotype.invalid:
            return
        if self.invalid:
            self.error = 100
            return
        try:
            if self.error is None:
                self.error = 0
            self.error += self.train_step(phenotype, images)
            self.last_error = self.error
            self.trained_samples += len(images)
            self.genome.increase_usage_counter()
        except Exception as err:
            traceback.print_exc()
            logger.error(err)
            logger.error(self.model)
            self.error += 100  # penalty for invalid genotype

    def do_eval(self, phenotype, images):
        self.eval_step(phenotype, images)
        # if self.invalid:
        #     self.error = 100
        #     return
        # self.error = self.error or 0
        # self.error += self.eval_step(phenotype, images)

    def create_model(self, input_data):
        self.input_shape = input_data.size()
        if "model" not in self._modules:
            self.model = self.transform_genotype(input_data)
        if self.optimizer is None:
            self.optimizer = self.genome.optimizer_gene.create_phenotype(self)
        self.genome.num_params = self.num_params()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def transform_genotype(self, input_data):
        """Generates a generic model using pytorch."""
        layers = []

        genes = list(self.genome.genes)
        genes += self.genome.output_genes  # add output layers

        # count how many deconvs exists in the model
        deconv_layers = list(filter(lambda x: isinstance(x, Deconv2d), genes))
        n_deconv2d = len(deconv_layers)
        has_new_gene = len([g for g in genes if g.used == 0]) > 0

        # iterate over genes to create a pytorch sequential model
        for i, gene in enumerate(genes):
            # TODO: move this code into the Genome class
            if i + 1 < len(genes):  # link the current gene with the next
                gene.next_layer = genes[i+1]
            if i > 0:
                gene.previous_layer = genes[i-1]

            next_input_size, next_input_shape = self.calc_output_size(layers, input_data)

            # adjust shape for linear layer
            if gene.is_linear() and len(next_input_shape) > 2:
                layers.append(Reshape((-1, next_input_size)))

            # adjust out_features of the last linear layer
            if isinstance(gene, Linear) and gene.is_last_linear():
                if isinstance(self.output_size, int):
                    gene.out_features = self.output_size
                else:
                    # TODO: check the dimensions of the first linear layer in generators
                    div = 2**n_deconv2d
                    if gene.next_layer is not None and not gene.next_layer.out_channels:
                        gene.next_layer.out_channels = 2 ** np.random.randint(config.layer.conv2d.min_channels_power, config.layer.conv2d.max_channels_power)
                    if isinstance(gene.next_layer, Deconv2d):
                        d = 2 if gene.next_layer.out_channels < 2 ** config.layer.conv2d.max_channels_power else 1
                        gene.out_features = d * gene.next_layer.out_channels * int(np.ceil(self.output_size[1] / div)) * int(np.ceil(self.output_size[2] / div))
                    else:
                        # should never occurs for generators
                        gene.out_features = 4 * int(np.ceil(self.output_size[1] / div)) * div * int(np.ceil(self.output_size[2] / div)) * div

            # adjust shape for 2d layer
            if not gene.is_linear() and len(next_input_shape) == 2:
                # adjust shape based on the next layers
                div = 2**n_deconv2d
                w, h = int(np.ceil(self.output_size[1]/div)), int(np.ceil(self.output_size[2]/div))
                next_input_shape = (-1, next_input_size//w//h, w, h)
                layers.append(Reshape(next_input_shape))

            new_layer = gene.create_phenotype(next_input_shape, self.output_size)
            if gene.used > 0 and has_new_gene:
                gene.freeze()
            else:
                gene.unfreeze()
            gene.module_name = "model.%d" % len(layers)
            layers.append(new_layer)

        return nn.Sequential(*layers)

    def calc_output_size(self, layers, input_data):
        current_model = nn.Sequential(*layers)
        current_model.eval()
        forward_pass = current_model(input_data)
        # return the product of the vector array (ignoring the batch size)
        return int(np.prod(forward_pass.size()[1:])), forward_pass.size()

    def set_requires_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value

    def fitness(self):
        if config.evolution.fitness.generator == "random":
            if self.random_fitness is None:
                self.random_fitness = np.random.rand()
            return self.random_fitness
        return self.error

    def save(self, path):
        torch.save(self.cpu(), path)

    def valid(self):
        return not self.invalid and self.error is not None

    def skill_rating_enabled(self):
        return config.stats.calc_skill_rating or config.evolution.fitness.discriminator == "skill_rating" or\
               config.evolution.fitness.generator == "skill_rating"

    def calc_skill_rating(self, adversarial):
        if not self.skill_rating_enabled():
            return
        rating = self.glicko.create_rating(mu=adversarial.skill_rating.mu, phi=adversarial.skill_rating.phi, sigma=adversarial.skill_rating.sigma)
        # self.skill_rating_games.append((1 if self.win_rate > 0.5 else 0, rating))
        self.skill_rating_games.append((self.win_rate, rating))

    def finish_calc_skill_rating(self):
        if not self.skill_rating_enabled():
            return
        if len(self.skill_rating_games) == 0:
            logger.warning("no games to update the skill rating")
            return
        print("finish_calc_skill_rating", self.skill_rating_games)
        self.skill_rating = self.glicko.rate(self.skill_rating, self.skill_rating_games)
        self.skill_rating_games = []

    def reset_optimizer_state(self):
        self.optimizer = self.genome.optimizer_gene.create_phenotype(self)

    @classmethod
    def load(cls, path):
        return torch.load(path, map_location="cpu")

    def __repr__(self):
        output_genes_str = " -> ".join([str(g) for g in self.genome.output_genes])
        return self.__class__.__name__ + f"(genome={self.genome}, output_layers={output_genes_str})"

    def to_json(self):
        """Create a json representing the model"""
        ret = []
        for gene in self.genome.genes + self.genome.output_genes:
            d = dict(gene.__dict__)
            del d["uuid"], d["module"], d["next_layer"], d["previous_layer"], d["normalization"], d["wscale"]
            ret.append({
                "type": gene.__class__.__name__,
                "wscale": gene.has_wscale(),
                "minibatch_stddev": gene.has_minibatch_stddev(),
                "normalization": gene.normalization.__class__.__name__ if gene.normalization is not None else None,
                **d
            })
        return json.dumps(ret)
