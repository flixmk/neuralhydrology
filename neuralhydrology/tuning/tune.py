import os
import yaml
import logging
from pathlib import Path
from datetime import datetime

import neuralhydrology
from neuralhydrology.utils.config import Config
from neuralhydrology.tuning.nh_tuner import Nh_Tuner
from neuralhydrology.utils.logging_utils import setup_logging
from distutils.command.config import config

import ray
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

LOGGER = logging.getLogger(__name__)

class HpTuner():

    def __init__(self):
        self.possible_hyperparameters = {"hidden_size": (23, 256, int),
                                         "output_dropout": (0.0, 0.99, float), 
                                         "initial_forget_bias": (0.0, 7.0, float),
                                         "model": ("all", str)}
        self.default_num_runs_per_gpu = 1
        self.default_num_runs_per_cpu = 1
    
    def save_settings(self, config_file, search_space):
        """
        Function to allow communication between neuralhydrology and ray tune. (Trainable class)
        Creates a .yml file in the installation directory. Stupid solution...
        """
        nh_dir = neuralhydrology.__path__[0]
        Path(nh_dir + "/tuning/settings/").mkdir(parents=True, exist_ok=True)
        settings = {"yml_config": str(config_file.resolve()), 
                    "wd": os.getcwd(),
                    "params": list(self.possible_hyperparameters.keys())}
        with open(rf"{nh_dir}/tuning/settings/settings.yml", 'w') as file:
            yaml.dump(settings, file)
            
    def read_search_space(self, cfg):
        
        settings = cfg.hptuning["settings"]
        search_space = dict()
        for key, _ in self.possible_hyperparameters.items():
            if key in settings:
                if self.possible_hyperparameters[key][-1] == int:
                    search_space[key] = tune.randint(settings[key][0], settings[key][1])
                elif self.possible_hyperparameters[key][-1] == float:
                    search_space[key] = tune.uniform(settings[key][0], settings[key][1])
                elif self.possible_hyperparameters[key][-1] == str:
                    if settings[key][0] == "all":
                        search_space[key] = tune.choice(["cudalstm", "gru"])
                    else:
                        search_space[key] = tune.choice([settings[key][i] for i in range(len(settings[key]))])
        return search_space
    
    def run_tuning(self, config_file):
        """_summary_

        Args:
            config_file (Path): Standard Config file.
            method (string): Select one of the following methods: tpe, cma-es, ... Defaults to None -> tpe
            params (dict, tuple: (start, end, step_size [optional])): Dict of params. key contains name. value contains tuple. Defaults to None (use all with some 
            predefined ranges)
            n_runs (int, optional): How many iterations to go. Defaults to 100.
            model (list, optional): List that contains all models as default. Currently only the 3 lstms. Defaults to None.
        """

        setup_logging("./hptuning.log")
        
        cfg = Config(config_file)
        search_space = self.read_search_space(cfg)
        
        self.save_settings(config_file, search_space)
        
        num_samples = cfg.hptuning["runs"]
        
        if cfg.hptuning.get("sim_runs_per_gpu") is None:
            # preferred default setting for nr of runs per gpu at the same time
            # change the denominator to the number of parallel runs
            num_runs_per_gpu = self.default_num_runs_per_gpu
        elif isinstance(cfg.hptuning["sim_runs_per_gpu"], int):
            num_runs_per_gpu = 1/cfg.hptuning["sim_runs_per_gpu"]
        else:
            LOGGER.warn(f"Something is wrong with the \"sim_runs_per_gpu\" attribute")
        
        if cfg.hptuning.get("sim_runs_per_cpu") is None:
            # preferred default setting for nr of runs per cpu at the same time
            # change the denominator to the number of parallel runs
            num_runs_per_cpu = self.default_num_runs_per_cpu
        elif isinstance(cfg.hptuning["sim_runs_per_cpu"], int):
            num_runs_per_cpu = 1/cfg.hptuning["sim_runs_per_cpu"]
        else:
            LOGGER.warn(f"Something is wrong with the \"sim_runs_per_cpu\" attribute")
            
            
        search_alg = OptunaSearch()
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=10)
        self.tuner = tune.Tuner(
            tune.with_resources(
            Nh_Tuner,
            resources={"cpu":num_runs_per_cpu, "gpu": num_runs_per_gpu}
            ),
            tune_config=tune.TuneConfig(
                metric="mean_loss",
                mode="min",
                search_alg=search_alg,
                num_samples=num_samples,
            ),
            param_space=search_space,
            run_config=air.RunConfig(
                stop={"training_iteration": 1},
            ),
        )
        
        results = self.tuner.fit()
        
        
        LOGGER.info(f"Best hyperparameters found were: {results.get_best_result().config}")