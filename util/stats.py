import torch
import torchvision.utils as vutils
import pandas as pd
import numpy as np
import logging
from evolution.config import config, GAN_CONFIG_FILE
import os
import shutil
from util.notifier import notify
from datetime import datetime
from metrics import rmse_score
import socket

logger = logging.getLogger(__name__)


class Stats:

    def __init__(self, log_dir=None, input_shape=None, train_loader=None, validation_loader=None):
        self.test_noise = None
        self.input_shape = input_shape
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.last_notification = None
        self.log_dir = log_dir
        if not self.log_dir:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = os.path.join('runs', f"{current_time}_{socket.gethostname()}")
        os.makedirs(os.path.join(self.log_dir, "images"))
        # copy config file into log dir
        shutil.copy(os.path.join(os.path.dirname(__file__), "..", f"{GAN_CONFIG_FILE}.py"), self.log_dir)

    def save_data(self, epoch, g_pop, d_pop, save_best_model=False):
        noise_path = os.path.join(os.path.dirname(__file__), "..", "./generator_noise.pt")
        if not os.path.isfile(os.path.join(self.log_dir, "./generator_noise.pt")) and \
                os.path.exists(noise_path):
            shutil.copy(noise_path, self.log_dir)  # copy noise file into log dir

        epoch_dir = f"{self.log_dir}/generations/{epoch:03d}"
        os.makedirs(epoch_dir)
        global_data_values = {}
        for name, pop in [("d", d_pop), ("g", g_pop)]:
            phenotypes = pop.phenotypes()
            global_data_values[f"species_{name}"] = len(pop.species_list)
            global_data_values[f"speciation_threshold_{name}"] = pop.speciation_threshold
            global_data_values[f"invalid_{name}"] = sum([p.invalid for p in phenotypes])
            for k, v in pop.stats.items():
                global_data_values[f"{k}_{name}"] = v
            # generate data for current generation
            columns = ["loss", "trained_samples", "layers", "genes_used",
                       "model", "species_index", "fitness", "generation", "age", "parameters", "skill_rating"]
            if name == "g":
                columns.append("fid_score")
                columns.append("inception_score")
                columns.append("rmse_score")
            df = pd.DataFrame(index=np.arange(0, len(phenotypes)), columns=columns)
            j = 0
            for i, species in enumerate(pop.species_list):
                for p in species:
                    values = [p.error, p.trained_samples, len(p.genome.genes), np.mean([g.used for g in p.genome.genes]),
                              p.to_json(), i, p.fitness(), p.genome.generation, p.genome.age, p.genome.num_params,
                              p.skill_rating.mu if p.skill_rating is not None else 0]
                    if name == "g":
                        values.append(p.fid_score)
                        values.append(p.inception_score_mean)
                        values.append(p.rmse_score)
                    df.loc[j] = values
                    j += 1
            df.sort_values('fitness').reset_index(drop=True).to_csv(f"{epoch_dir}/data_{name}.csv")

        # generate image for each G
        os.makedirs(f"{epoch_dir}/images")
        for i, g in enumerate(g_pop.sorted()):
            if not g.valid():
                continue
            self.generate_image(g, path=f"{epoch_dir}/images/generator-{i:03d}.png")
            if i == 0 and save_best_model:
                g.save(f"{epoch_dir}/generator.pkl")

        if save_best_model:
            d_pop.sorted()[0].save(f"{epoch_dir}/discriminator.pkl")

        # append values into global data
        global_data = pd.DataFrame(data=global_data_values, index=[epoch])
        data_file = f"{self.log_dir}/generations/data.csv"
        if os.path.exists(data_file):
            global_data = pd.read_csv(data_file).append(global_data, ignore_index=True)
        global_data.to_csv(data_file, index=False)

    def generate_image(self, G, path=None):
        if not G.valid():
            return None
        test_images = G(self.test_noise).detach()
        grid_images = [torch.from_numpy((test_images[k, :].data.cpu().numpy().reshape(self.input_shape) + 1)/2)
                        for k in range(config.stats.num_generated_samples)]
        grid = vutils.make_grid(grid_images, normalize=False, nrow=int(config.stats.num_generated_samples**(1/2)))
        # store grid images in the run folder
        if path is not None:
            vutils.save_image(grid, path)
        return grid

    def generate(self, g_pop, d_pop, epoch):
        num_epochs = config.evolution.max_generations
        if epoch % config.stats.print_interval != 0 and epoch != num_epochs - 1:
            return

        G = g_pop.best()
        D = d_pop.best()
        G.eval()
        D.eval()

        # this should never occurs!
        if G.invalid or D.invalid:
            logger.error("invalid D or G")
            return

        if self.test_noise is None:
            self.test_noise = G.generate_noise(config.stats.num_generated_samples, volatile=True).cpu()

        if config.stats.calc_fid_score_best and G.fid_score is None:
            G.calc_fid()

        if config.stats.calc_rmse_score:
            rmse_score.initialize(self.train_loader, config.evolution.fitness.fid_sample_size)
            for g in g_pop.phenotypes():
                g.calc_rmse_score()

        if config.stats.calc_inception_score:
            for g in g_pop.phenotypes():
                g.inception_score()

        self.save_data(epoch, g_pop, d_pop, config.stats.save_best_model and (epoch == num_epochs-1 or (epoch + 1) % config.stats.save_best_interval == 0))

        logger.debug("\n%s: D: %s G: %s", epoch, D.error, G.error)
        logger.debug(G)
        logger.debug(G.model)
        logger.debug(D)
        logger.debug(D.model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug((f"memory_allocated: {torch.cuda.memory_allocated()}, "
                          f"max_memory_allocated: {torch.cuda.max_memory_allocated()}, "
                          f"memory_cached: {torch.cuda.memory_cached()}, "
                          f"max_memory_cached: {torch.cuda.max_memory_cached()}"))

        if config.stats.notify and \
                (self.last_notification is None or
                 (datetime.now() - self.last_notification).seconds//60 > config.stats.min_notification_interval):
            self.last_notification = datetime.now()
            notify(f"Epoch {epoch}: G {G.fitness():.2f}, D: {D.error:.2f}")
