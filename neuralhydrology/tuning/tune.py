import os
import yaml
import logging
from pathlib import Path
from datetime import datetime

import neuralhydrology
from neuralhydrology.utils.config import Config
from neuralhydrology.tuning.nh_tuner import Nh_Tuner
from neuralhydrology.utils.logging_utils import setup_logging

from ray import tune, air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import CombinedStopper,MaximumIterationStopper, TrialPlateauStopper


LOGGER = logging.getLogger(__name__)

class HpTuner():

    def __init__(self, run_name = None):
        """Default values for all settings.
        If a new hyperparameters gets added, then just add them here and in the yml.
        """
        
        # used for tensorboard to find the location of the trial.
        self.run_name = run_name
        
        self.possible_hyperparameters = {"hidden_size": (23, 256, int),
                                         "output_dropout": (0.0, 0.99, float), 
                                         "initial_forget_bias": (0.0, 7.0, float),
                                         "batch_size": (32, 512, int),
                                         "model": ("all", str)}
        self.all_models = ["cudalstm", "gru"]
        self.metric_modes = {
            "NSE": "max", 
            "RMSE": "min", 
            "MSE": "min", 
            "KGE": "max",
            "Alpha-NSE": "max",
            "Pearson-r": "max",
            "Beta-KGE": "min",
            "Beta-NSE": "min",
            "FHV": "max",
            "FMS": "max",
            "FLV": "max",
            "Peak-Timing": "min"}
        self.default_num_runs_per_gpu = 1
        self.default_num_runs_per_cpu = 1
        self.default_metric = "NSE"
        
    
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
                        search_space[key] = tune.choice(self.all_models)
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
            num_runs_per_gpu = self.default_num_runs_per_gpu
        elif isinstance(cfg.hptuning["sim_runs_per_gpu"], int):
            num_runs_per_gpu = 1/cfg.hptuning["sim_runs_per_gpu"]
        else:
            LOGGER.warn(f"Something is wrong with the \"sim_runs_per_gpu\" attribute")
            
        if cfg.hptuning.get("sim_runs_per_cpu") is None:
            num_runs_per_cpu = self.default_num_runs_per_cpu
        elif isinstance(cfg.hptuning["sim_runs_per_cpu"], int):
            num_runs_per_cpu = 1/cfg.hptuning["sim_runs_per_cpu"]
        else:
            LOGGER.warn(f"Something is wrong with the \"sim_runs_per_cpu\" attribute")    
            
        if cfg.hptuning.get("metric") is None:
            metric = self.default_metric
        elif isinstance(cfg.hptuning["metric"], str):
            metric = cfg.hptuning["metric"]
            mode = self.metric_modes[metric]
        else:
            LOGGER.warn(f"Something is wrong with the \"metric\" attribute")    
            
        search_alg = OptunaSearch()
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=10)
        
        stopper = CombinedStopper(
            MaximumIterationStopper(max_iter=cfg.epochs),
            TrialPlateauStopper(metric=metric, mode=self.metric_modes[metric])
            )
        
        self.tuner = tune.Tuner(
            tune.with_resources(
            Nh_Tuner,
            resources={"cpu":num_runs_per_cpu, "gpu": num_runs_per_gpu}
            ),
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                search_alg=search_alg,
                num_samples=num_samples,
            ),
            param_space=search_space,
            run_config=air.RunConfig(
                name=self.run_name if self.run_name is not None else None,
                stop=stopper,
            ),
        )
        results = self.tuner.fit()
        LOGGER.info(f"Best hyperparameters found were: {results.get_best_result().config}")

        cfg_dict = cfg.as_dict()
        for key, value in results.get_best_result().config.items():
            cfg_dict[key] = value
        cfg._cfg = cfg_dict
        date_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        cfg.dump_config(folder=Path("./"), filename=f"{date_time}_best_config.yml")