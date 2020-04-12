#!/usr/bin/env python
# References:
# https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
# https://www.cs.ucf.edu/~kstanley/neat.html
# https://github.com/CodeReclaimers/neat-python/blob/99da17d4bd71ec97d7f37c9b5df0006c7689a893/neat/reproduction.py

# https://github.com/pytorch/examples/blob/master/dcgan/main.py
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from evolution.discriminator import Discriminator
from evolution.generator import Generator
from util.stats import Stats
import numpy as np
from tqdm import tqdm
from .config import config
import logging
from util.folder import ImageFolder
from metrics import generative_score
import os
from .neat import NEAT
from .nsga2 import NSGA2
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class GanTrain:

    def __init__(self, log_dir=None):
        full_dataset = self.create_dataset()
        train_len = int(0.9 * len(full_dataset))
        train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_len, len(full_dataset) - train_len])
        logger.info("train size: %d, validation size: %d" % (len(train_dataset), len(validation_dataset)))
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.gan.batch_size, num_workers=config.gan.data_loader_workers, drop_last=True,
            shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=config.gan.batch_size, num_workers=config.gan.data_loader_workers,
            drop_last=True, shuffle=True)
        self.input_shape = next(iter(self.train_loader))[0].size()[1:]
        self.stats = Stats(log_dir=log_dir, input_shape=self.input_shape, train_loader=self.train_loader,
                           validation_loader=self.validation_loader)
        evaluator = Evaluator(self.train_loader, self.validation_loader)
        self.evolutionary_algorithm = {"NEAT": NEAT, "NSGA2": NSGA2}[config.evolution.algorithm](evaluator)

    @classmethod
    def create_dataset(cls):
        transform_arr = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        if config.gan.dataset_resize:
            transform_arr.insert(0, transforms.Resize(config.gan.dataset_resize))
        transform = transforms.Compose(transform_arr)
        base_path = os.path.join(os.path.dirname(__file__), "..", "data")
        if hasattr(dsets, config.gan.dataset):
            dataset = getattr(dsets, config.gan.dataset)(
                root=os.path.join(base_path, config.gan.dataset),
                download=True, transform=transform)
            if config.gan.dataset_classes:
                indexes = np.argwhere(np.isin(dataset.targets, config.gan.dataset_classes))
                dataset.data = dataset.data[indexes].squeeze()
                dataset.targets = np.array(dataset.targets)[indexes]
            return dataset
        else:
            return ImageFolder(root=os.path.join(base_path, config.gan.dataset, "train"), transform=transform)

    def start(self):
        if config.evolution.fitness.generator == "FID" or config.stats.calc_fid_score or config.stats.calc_fid_score_best:
            generative_score.initialize_fid(self.train_loader, sample_size=config.evolution.fitness.fid_sample_size)

        generators_population = self.evolutionary_algorithm.intialize_population(config.gan.generator.population_size, Generator, output_size=self.input_shape)
        discriminators_population = self.evolutionary_algorithm.intialize_population(config.gan.discriminator.population_size, Discriminator, output_size=1, input_shape=[1] + list(self.input_shape))
        # initial evaluation
        self.evolutionary_algorithm.evaluate_population(generators_population.phenotypes(), discriminators_population.phenotypes())
        for generation in tqdm(range(config.evolution.max_generations-1)):
            self.stats.generate(generators_population, discriminators_population, generation)
            # executes selection, reproduction and replacement to create the next population
            generators_population, discriminators_population = self.evolutionary_algorithm.compute_generation(generators_population, discriminators_population)
        # stats for last generation
        self.stats.generate(generators_population, discriminators_population, generation+1)
