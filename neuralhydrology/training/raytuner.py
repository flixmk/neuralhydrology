from distutils.command.config import config
import logging
import os
# optimization libraries
import optuna
# pip install git+https://github.com/subpath/neuro-evolution.git


import neuralhydrology
from pathlib import Path
from datetime import datetime
from neuralhydrology.utils.config import Config
import torch
import numpy as np
from neuralhydrology.utils.logging_utils import setup_logging

from neuralhydrology.nh_run import start_hptuning

import yaml


import ray
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch


LOGGER = logging.getLogger(__name__)
info_var = None

# TODO: Make hparams optimize a custom goal
# TODO: What to do with learning rate? LR Scheduler?
class Settings():
    
    def __init__(self, config, config_file):
        self.yml_config = config
        self.yml_config_PATH = config_file

# Highest interaction level
class HpTuner():
    """_summary_
    """
    def __init__(self):
        self.method = 'bayesian'
        self.possible_methods = ['bayesian', 'cma-es', 'tpe', 'pbt']
        self.tuner = None
        self.config_file = None
        
    def create_settings(self): 
        global info_var
        info_var = Settings(self.config, self.config_file)
        
    def save_settings(self):
        nh_dir = neuralhydrology.__path__[0]
        Path(nh_dir + "/hptuning_settings/").mkdir(parents=True, exist_ok=True)
        settings = {"yml_config": str(self.config_file.resolve()), "wd": os.getcwd()}

        with open(rf"{nh_dir}/hptuning_settings/settings.yml", 'w') as file:
            documents = yaml.dump(settings, file)
        
        
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

        #self.tuner = RayTuner()

        #self.tuner.run(config_file)
        
        LOGGER.info("Started running!")
        
        self.config_file = config_file
        self.config = Config(config_file)
        
        self.create_settings()
        self.save_settings()
        # LOGGER.info(f"INFO_VAR: {info_var}")
        
        self.search_space = {
            "hidden_size": tune.randint(32, 256),
            "output_dropout": tune.uniform(0.0, 0.9),
            "model": tune.choice(['cudalstm'])
        }
        
        self.working_dir = str(Path().absolute())
        
        config_dict = self.config.as_dict()
        # print(config_dict)
        
        algo = OptunaSearch()
        algo = ConcurrencyLimiter(algo, max_concurrent=2)
        num_samples = 3
        # print("TORCH_GPU: ",torch.cuda.is_available())
        # ray.init(num_gpus=1)
        # print(ray.get_gpu_ids())
        # tune.utils.wait_for_gpu()
        self.tuner = tune.Tuner(
            tune.with_resources(
            RayTuner,
            resources={"cpu":1, "gpu": 1}
        ),
            tune_config=tune.TuneConfig(
                metric="mean_loss",
                mode="min",
                search_alg=algo,
                num_samples=num_samples,
            ),
            param_space=self.search_space,
            run_config=air.RunConfig(
        stop={"training_iteration": 1},
    ),
        )
        
        results = self.tuner.fit()
        
        
        LOGGER.info(f"Best hyperparameters found were: {results.get_best_result().config}")




class RayTuner(tune.Trainable):
    """_summary_

    Args:
        Tuner (_type_): _description_
    """

    # def __init__(self):
    #     super(RayTuner, self).__init__()
    #     LOGGER.info(f"Using RAYTUNE")
    #     self.config = None
    #     self.param_names_range = None
    #     self.params = None
    #     self.run_id = 0
        

        
        
    def setup(self, tuning_config):
        nh_dir = neuralhydrology.__path__[0]

        with open(rf"{nh_dir}/hptuning_settings/settings.yml") as file:
            settings = yaml.load(file, Loader=yaml.FullLoader)
        print("SETTINGS:", settings)
        print("CWD:", os.getcwd())
        
        self.yml_config_PATH = Path(settings['yml_config'])
        self.yml_config = Config(self.yml_config_PATH)
        self.working_dir = settings["wd"]
        
        
        self.search_space = {
            "hidden_size": tune.randint(32, 256),
            "output_dropout": tune.uniform(0.0, 0.9),
            "model": tune.choice(['cudalstm'])
        }
        
        self.tuning_config = tuning_config
        self.hidden_size = tuning_config["hidden_size"]
        self.output_dropout = tuning_config["output_dropout"]
        self.model = tuning_config["model"]
    
    def step(self):
        score = self.objective(self.tuning_config)
        return score
    
    def save_checkpoint(self, checkpoint_dir: str):
        pass
    def load_checkpoint(self, checkpoint_dir: str):
        pass

    def define_hparams(self, tuning_config):
        config_dict = self.yml_config.as_dict()
        
        ## setting the hyperparameters
        print("The following parameters get used this run:")
        for key, value in self.search_space.items():
            config_dict[key] = tuning_config[key]
            LOGGER.info(f"{key}, {config_dict[key]}")
        self.yml_config._cfg = config_dict
        
        
        dt = datetime.now() 
        file_name = "hp--" + str(dt).replace(":", '_').replace('.','_') + ".yml"
        
        print(self.working_dir + "/yml_folder/"+str(file_name))
        print(self.yml_config)
        self.yml_config.dump_config(Path(self.working_dir + "/yml_folder/"), filename=file_name)
        self.config_file = Path(self.working_dir + "/yml_folder/" + str(file_name))

    def objective(self, tuning_config):
        print("GOT HERE TO OBJECTIVE")
        
        print("TORCH_GPU: ",torch.cuda.is_available())
        self.define_hparams(tuning_config)
        os.chdir(self.working_dir)
        if torch.cuda.is_available():
            print("Using GPU")
            metrics = start_hptuning(config_file=self.yml_config_PATH)

        # fall back to CPU-only mode
        else:
            metrics = start_hptuning(config_file=self.yml_config_PATH, gpu=-1)

        eval_metric = self.get_metrics(metrics, 'avg_loss',3)
        return {"mean_loss": eval_metric}



    def get_metrics(self, metrics, used_metric,epoch_lookback):
        # epoch_lookback: average the last x epochs for a metric to not just use a lucky drop in loss.
        sorted_metrics = dict()
        for key in metrics[0].keys():
            sorted_metrics[key] = list()
        for metrics_epoch in metrics:
            for key in metrics_epoch.keys():
                sorted_metrics[key].append(metrics_epoch[key])
        eval_metric = sorted_metrics[used_metric][-epoch_lookback:]
        eval_metric = np.mean(np.array(eval_metric))

        LOGGER.info(f"Using {used_metric} as metric.")
        LOGGER.info(f"Averaging the last {epoch_lookback} epochs.")

        return eval_metric
    
    



    # def run(self, config_file: Path):
        
    #     LOGGER.info("Started running!")
        
    #     self.config_file = config_file
    #     self.config = Config(config_file)
        
    #     self.working_dir = str(Path().absolute())
        
    #     config_dict = self.config.as_dict()
    #     print(config_dict)
        
    #     algo = OptunaSearch()
    #     algo = ConcurrencyLimiter(algo, max_concurrent=2)
    #     num_samples = 1000
    #     print("TORCH_GPU: ",torch.cuda.is_available())
        
    #     # ray.init(num_gpus=1)
    #     # print(ray.get_gpu_ids())
    #     # tune.utils.wait_for_gpu()
    #     self.tuner = tune.Tuner(
    #         tune.with_resources(
    #         Trainable_Tuner(1),
    #         resources={"cpu":1, "gpu": 1}
    #     ),
    #         tune_config=tune.TuneConfig(
    #             metric="mean_loss",
    #             mode="min",
    #             search_alg=algo,
    #             num_samples=num_samples,
    #         ),
    #         param_space=self.search_space,
    #     )
        
    #     results = self.tuner.fit()
        
        
    #     LOGGER.info(f"Best hyperparameters found were: {results.get_best_result().config}")
        
