params = {
    "evolution": {
        "max_generations": 50,
        "elitism": 0.0,
        "tournament_size": 2,
        "max_layers": 3,
        "sequential_layers": False,  # when true, genes/layers will be added sequentially
        "add_layer_prob": 0.1,
        "rm_layer_prob": 0.1,
        "gene_mutation_prob": 0.3,
        "crossover_rate": 0.0,
        "conv_beginning": False,
        "freeze_when_change": False,
        "control_number_layers": False,
        "freeze_best": False,
        "algorithm": "NSGA2",  # NEAT | NSGA2
        "nslc": {
            "archive_prob": 0.1,
            "neighbors_size": 3  # number or "all"
        },
        "evaluation": {
            "type": "all-vs-all",  # all-vs-all | random | all-vs-best | all-vs-kbest | all-vs-species-best | spatial
            "best_size": 0,
            "iterations": 1,
            "initialize_all": True,  # apply all-vs-all for the first time type is all-vs-best, all-vs-kbest or all-vs-species-best
            "reset_optimizer": False
        },
        "speciation": {
            "size": 2,
            "keep_best": True,
            "threshold": 1,
            "distance": "num_genes"  # parameter | uuid | num_genes | num_genes_per_type
        },
        "fitness": {
            "discriminator": "skill_rating",  # AUC | loss | random | skill_rating
            "generator": "skill_rating",  # AUC | FID | loss | random | skill_rating
            "fid_sample_size": 256,  # TODO: this is also the sample size for the inception score
            "fid_dimension": 2048,  # 2048 | 768 | 192 | 64
            "fid_batch_size": 32,
            "evaluation_batches": 1,
            "skill_rating": {
                "tau": 0.1,
                "sigma": 0.06
            }
        }
    },
    "gan": {
        "dataset": "MNIST",  # MNIST | FashionMNIST | CIFAR10 | CelebA | SVHN
        "dataset_resize": None,  # None to disable, [W, H] to resize
        "dataset_classes": None,  # select a subset of classes to use (set to None to use all)
        "batches_limit": 50,
        "batch_size": 64,
        "data_loader_workers": 1,
        "critic_iterations": 1,
        "type": "gan",  # gan | wgan | lsgan | rsgan | rasgan
        "label_smoothing": False,  # . not significant
        "batch_normalization": True,  # +++ much better
        "pixelwise_normalization": False,
        "use_wscale": False,  # -- worse
        "use_minibatch_stddev": False,  # + slightly better
        "dropout": False,  # --- worse
        "discriminator": {
            "population_size": 5,
            "simple_layers": False,
            "fixed": False,
            "use_gradient_penalty": False,  # - slightly worse
            "gradient_penalty_lambda": 10,
            "possible_layers": ["Conv2d"],
            "optimizer": {
                "type": "Adam",  # Adam | RMSprop | SGD | Adadelta
                "copy_optimizer_state": False,
                "learning_rate": 5e-4,
                "weight_decay": 0,
            },
        },
        "generator": {
            "population_size": 5,
            "simple_layers": False,
            "fixed": False,
            "possible_layers": ["Deconv2d"],
            "optimizer": {
                "type": "Adam",  # Adam | RMSprop | SGD | Adadelta
                "copy_optimizer_state": False,
                "learning_rate": 5e-4,
                "weight_decay": 0,
            },
        },
    },
    "layer": {
        "keep_weights": True,
        "resize_weights": True,
        "resize_linear_weights": True,
        "activation_functions": ["ReLU", "LeakyReLU"],
        "linear": {
            "min_features_power": 5,
            "max_features_power": 8  # max number of channels is 2**max_channels_power
        },
        "conv2d": {
            "min_channels_power": 5,
            "max_channels_power": 8,  # max number of channels is 2**max_channels_power
            "random_out_channels": True,  # when false, will calculate output_channels based on in_channels
            "kernel_size": 4,
            "force_double": False
        }
    },
    "stats": {
        "num_generated_samples": 36,
        "print_interval": 1,
        "calc_inception_score": False,
        "calc_fid_score": False,
        "calc_fid_score_best": False,
        "calc_rmse_score": False,
        "calc_skill_rating": False,
        "save_best_model": True,
        "save_best_interval": 5,
        "notify": False,
        "min_notification_interval": 30
    }
}
