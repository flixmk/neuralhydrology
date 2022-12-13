import logging
import os
import neuralhydrology
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.umaltrainer import UMALTrainer
import torch
import numpy as np
from neuralhydrology.training.train import start_tuning
import yaml
from ray import tune

LOGGER = logging.getLogger(__name__)

class Nh_Tuner(tune.Trainable):

    def setup(self, tuning_config):
        """Gets called for each new trial once. 

        Args:
            tuning_config (_type_): _description_
        """
        self.load_settings()
        self.tuning_config = tuning_config
        # define hyperparameter of this trial
        self.define_hparams(tuning_config)
        
        # ray tune changes the working directory.
        # need to change it back to folder with .yml
        os.chdir(self.working_dir)
        if self.cfg.head.lower() in ['regression', 'gmm', 'cmal', '']:
            self.trainer = BaseTrainer(cfg=self.cfg, logging_flag=False)
        elif self.cfg.head.lower() == 'umal':
            self.trainer = UMALTrainer(cfg=self.cfg)
        else:
            raise ValueError(f"Unknown head {self.cfg.head}.")
        self.trainer.initialize_training()
        self.epoch_iteration = 0
        
    def step(self):
        score = self.objective(self.tuning_config)
        return score
    
    def save_checkpoint(self, checkpoint_dir: str):
        pass
    def load_checkpoint(self, checkpoint_dir: str):
        pass
    
    def load_settings(self):
        """Method to communicate between closed raytune environment and neuralhydrology
        """
        nh_dir = neuralhydrology.__path__[0]
        # hard-coded settings file in the install directory
        with open(rf"{nh_dir}/tuning/settings/settings.yml") as file:
            settings = yaml.load(file, Loader=yaml.FullLoader)
        self.cfg_PATH = Path(settings['yml_config'])
        self.cfg = Config(self.cfg_PATH)
        self.working_dir = settings["wd"]
        self.possible_hyperparameters = settings["params"]

    def define_hparams(self, tuning_config):
        """Creating a Config with the proposed hyperparameters.

        Args:
            tuning_config (_type_): _description_
        """
        config_dict = self.cfg.as_dict()
        ## setting the hyperparameters
        for key, value in self.possible_hyperparameters.items():
            if value[-1] == float:
                config_dict[key] = round(tuning_config[key],3)
            else:
                config_dict[key] = tuning_config[key]
        ## setting metric to be optimized
        if "metric" in config_dict["hptuning"].keys():
            config_dict["metrics"] = [config_dict["hptuning"]["metric"]]
            
        self.cfg._cfg = config_dict
    

    def objective(self, tuning_config):
        """Main function of the tuning. Includes training and evaluation. 
        Missing stepwise operation.

        Args:
            tuning_config (_type_): _description_

        Returns:
            float: metric
        """
        metrics = self.trainer.train_and_validate_iteration(epoch=self.epoch_iteration)
        self.epoch_iteration += 1
        
        
        eval_metric = self.get_metrics(metrics, self.cfg.hptuning["metric"])
        return {self.cfg.hptuning["metric"]: eval_metric}

    def get_metrics(self, metrics, metric_name, epoch_lookback=1, logging=False):
        """_summary_

        Args:
            metrics (_type_): _description_
            metric_name (_type_): _description_
            epoch_lookback (int, optional): How many of the last epochs are going to be averaged and compared to get a better hint to a consistent improvement.. Defaults to 1.
            logging (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # epoch_lookback: average the last x epochs for a metric to not just use a lucky drop in loss.
        sorted_metrics = dict()
        for key in metrics[0].keys():
            sorted_metrics[key] = list()
            
        for metrics_epoch in metrics:
            for key in metrics_epoch.keys():
                sorted_metrics[key].append(metrics_epoch[key])
                
        eval_metric = sorted_metrics[metric_name][-epoch_lookback:]
        eval_metric = np.mean(np.array(eval_metric))
        LOGGER.info(f"Using {metric_name} as metric.") if logging else None
        LOGGER.info(f"Averaging the last {epoch_lookback} epochs.") if logging else None
        return eval_metric
    
    def start_run_with_metrics(self, cfg: Config, gpu: int = None):
        """modified method to start the training without saving a .yml and with returning the metrics.

        Args:
            cfg (Config): _description_
            gpu (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # check if a GPU has been specified as command line argument. If yes, overwrite config
        if gpu is not None and gpu >= 0:
            cfg.device = f"cuda:{gpu}"
        if gpu is not None and gpu < 0:
            cfg.device = "cpu"
        metrics = start_tuning(cfg)
        return metrics