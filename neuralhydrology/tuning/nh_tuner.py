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
from neuralhydrology.training.train import start_tuning


import yaml


import ray
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch


LOGGER = logging.getLogger(__name__)

# TODO: Files abändern beim hyperparameter tuning so dass auch hyperparameter drin sind (keine überschneidungen mehr!) DONE
# TODO: Logging to unterdrücken? Create lookup table unterdrücken?
# TODO: PBAR unterdrücken 
# TODO: LOGGING schauen obs funktioniert: FUNKTIONIERT
# TODO: Metricen als objective fürs tuning.
# TODO: mclstm doesnt work / embcudalstm deprecated / arlstm doesnt work / ealstm doesnt work !!!
# TODO: Find second logger for hptuning and disable it.

class Nh_Tuner(tune.Trainable):

    def setup(self, tuning_config):
        
        self.load_settings()
        
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
    
    def load_settings(self):
        nh_dir = neuralhydrology.__path__[0]

        with open(rf"{nh_dir}/tuning/settings/settings.yml") as file:
            settings = yaml.load(file, Loader=yaml.FullLoader)
        
        self.yml_config_PATH = Path(settings['yml_config'])
        self.yml_config = Config(self.yml_config_PATH)
        self.working_dir = settings["wd"]
        self.possible_hyperparameters = settings["params"]

    def define_hparams(self, tuning_config):
        config_dict = self.yml_config.as_dict()
        
        ## setting the hyperparameters
        for key in self.possible_hyperparameters:
            config_dict[key] = tuning_config[key]
        self.yml_config._cfg = config_dict
        dt = datetime.now() 
        file_name = "hp--" + str(dt).replace(":", '_').replace('.','_') + ".yml"
        self.yml_config.dump_config(Path(self.working_dir + "/yml_folder/"), filename=file_name)
        self.config_file = Path(self.working_dir + "/yml_folder/" + str(file_name))

    def objective(self, tuning_config):
        self.define_hparams(tuning_config)
        os.chdir(self.working_dir)
        if torch.cuda.is_available():
            metrics = self.start_run_with_metrics(config_file=self.config_file)
        # fall back to CPU-only mode
        else:
            metrics = self.start_run_with_metrics(config_file=self.config_file, gpu=-1)
        eval_metric = self.get_metrics(metrics, 'avg_loss',3)
        return {"mean_loss": eval_metric}

    def get_metrics(self, metrics, used_metric,epoch_lookback, logging=False):
        # epoch_lookback: average the last x epochs for a metric to not just use a lucky drop in loss.
        sorted_metrics = dict()
        for key in metrics[0].keys():
            sorted_metrics[key] = list()
        for metrics_epoch in metrics:
            for key in metrics_epoch.keys():
                sorted_metrics[key].append(metrics_epoch[key])
        eval_metric = sorted_metrics[used_metric][-epoch_lookback:]
        eval_metric = np.mean(np.array(eval_metric))

        LOGGER.info(f"Using {used_metric} as metric.") if logging else None
        LOGGER.info(f"Averaging the last {epoch_lookback} epochs.") if logging else None

        return eval_metric
    
    def start_run_with_metrics(self, config_file: Path, gpu: int = None):
        config = Config(config_file)

        # check if a GPU has been specified as command line argument. If yes, overwrite config
        if gpu is not None and gpu >= 0:
            config.device = f"cuda:{gpu}"
        if gpu is not None and gpu < 0:
            config.device = "cpu"
            
        metrics = start_tuning(config)
        return metrics