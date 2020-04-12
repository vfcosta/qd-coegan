import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import unittest
import shutil
from evolution.gan_train import GanTrain
from evolution.config import config


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.test_path = "/tmp/gan"
        shutil.rmtree(self.test_path, ignore_errors=True)
        os.makedirs(self.test_path, exist_ok=True)
        config.gan.generator.optimizer.copy_optimizer_state = True
        config.gan.generator.optimizer.type = "Adam"
        config.gan.discriminator.optimizer = config.gan.generator.optimizer
        config.evolution.sequential_layers = True
        config.evolution.max_generations = 3
        config.gan.batches_limit = 2
        config.gan.batch_size = 4
        config.gan.discriminator.population_size = 3
        config.gan.generator.population_size = 3
        config.layer.conv2d.max_channels_power = 5
        config.evolution.fitness.generator = "loss"
        config.evolution.fitness.fid_sample_size = 10
        config.evolution.algorithm = "NEAT"
        self.train = GanTrain(log_dir=self.test_path)

    def tearDown(self):
        shutil.rmtree(self.test_path, ignore_errors=True)

    def test_train(self):
        self.train.start()

    def test_fid(self):
        config.evolution.fitness.generator = "FID"
        self.train.start()

    def test_nsga2(self):
        shutil.rmtree(self.test_path, ignore_errors=True)
        config.evolution.algorithm = "NSGA2"
        self.train = GanTrain(log_dir=self.test_path)
        self.train.start()
