import logging
import os
# optimization libraries
import optuna
# pip install git+https://github.com/subpath/neuro-evolution.git

from pathlib import Path
from neuralhydrology.utils.config import Config
import torch
import numpy as np
from neuralhydrology.utils.logging_utils import setup_logging

from neuralhydrology.nh_run import start_hptuning

LOGGER = logging.getLogger(__name__)

# TODO: Make hparams optimize a custom goal
# TODO: What to do with learning rate? LR Scheduler?


# Highest interaction level
class HpTuner():
    """_summary_
    """
    def __init__(self):
        self.method = 'bayesian'
        self.possible_methods = ['bayesian', 'cma-es', 'tpe', 'pbt']
        self.tuner = None

        pass

    def select_method(self, method: str = None):
        # TODO: Maybe other baysian methods aswell? TPE seems to be the most prominent one there.
        if method is None or method == 'tpe':
            self.method = 0
            return OptunaTuner(sampler_id=self.method)
        elif method == 'cma-es':
            self.method = 1
            return OptunaTuner(sampler_id=self.method)
        elif method == 'bayesian':
            self.method = 0
            return OptunaTuner(sampler_id=self.method)
        elif method == 'pbt':
            self.method = 'pbt'
        else:
            LOGGER.error('The selected method is none of the offered optimization algorithms.\n'
                         'Try one of the following:\n'
                         '\'bayesian\' (default)\n'
                         '\'cma-es\'\n'
                         '\'tpe\'(Tree-structured Parzen Estimators)\n'
                         '\'pbt\'(Population-based Training)\n')

    def set_params(self, params: dict = None, model=None):
        self.tuner.set_params(params, model)

    def run_tuning(self, config_file, method=None, params=None, n_runs=100, model=['cudalstm','ealstm','mtslstm',]):
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

        self.tuner = self.select_method(method)
        self.set_params(params, model)

        self.tuner.run(config_file, n_runs)

    def evaluate_runs(self):
        pass

    def save_best(self):
        self.tuner.get_best_params()

# Super class for the tuners
class Tuner():
    """_summary_
    """
    def __init__(self):
        pass

    def get_possible_hparams(self):
        # TODO: Learningrate scheduler instead of decreasing the rate in steps? -> less hparams to tune.
        predefined_ranges = {
                                'hidden_size': (128, 512, 16),
                                'output_dropout': (0.0,0.9,0.1),
                                'initial_forget_bias': (0.0, 5.0, 0.5),
                                #'learning_rate': {
                                #    0: (0.01), 30: (0.005), 40: (0.001)
                                #}
                            }
        param_info = {
                        'hidden_size': 'int',
                        'output_dropout': 'float',
                        'initial_forget_bias': 'float',
                        #'learning_rate': 'float'
                    }
        


        LOGGER.info("The following hyperparameters are getting optimized:")
        LOGGER.info("Format: Name: (start, end, step size)")
        LOGGER.info(predefined_ranges)
        return predefined_ranges, param_info

    def run(self):
        # runs the tuning process for the respective method including the method calls from the used library
        pass

    def set_params(self):
        # setting the params that need to be optimized
        pass

    def define_hparams(self):
        # iterate over hparams
        pass

    def get_metrics(self):
        # method to get the parameters
        pass
    
    def get_model_info(self):
        model_info = [
                        'cudalstm', 
                        'ealstm', 
                        'mtslstm',             
                    ]
            
        return model_info


# specific class for each tuner
class OptunaTuner(Tuner):
    """_summary_

    Args:
        Tuner (_type_): _description_
    """

    def __init__(self, sampler_id):
        super(OptunaTuner, self).__init__()
        self.config = None
        self.study = None
        self.param_names_range = None
        self.params = None
        self.sampler_id = sampler_id
        self.predefined_ranges, self.param_info = self.get_possible_hparams()
        pass

    def set_params(self, params=None, model=None):
        if params == None:
            self.param_names_range = self.predefined_ranges
        else:
            self.param_names_range = params
            
        if model == None:
            self.model_info = self.get_model_info()
        else:
            self.model_info = model

    def define_hparams(self, trial):
        config_dict = self.config.as_dict()
        
        ## setting the hyperparameters
        LOGGER.info("The following parameters get used this run:")
        for key, value in self.param_names_range.items():
            # if no ranges were predefined by the user.
            if value == None:
                value = self.predefined_ranges[key]
            start = value[0]
            end = value[1]
            if len(value) == 2:
                step_size = self.predefined_ranges[key][2]
            else:
                step_size = value[2]
            if self.param_info[key] == 'int':
                config_dict[key] = trial.suggest_int(key, start, end, step=step_size)
            elif self.param_info[key] == 'float':
                config_dict[key] = trial.suggest_float(key, start, end, step=step_size)
            LOGGER.info(f"{key}, {config_dict[key]}")
        
        
        ## setting the model
        avail_models = self.model_info
        LOGGER.info(f"Available Models: {avail_models}")
        model_id = trial.suggest_int('model', 0, len(avail_models))
        config_dict['model'] = avail_models[model_id]
        LOGGER.info(f"Chosen model for this run: {avail_models[model_id]}")
            

        self.config._cfg = config_dict

        if os.path.exists("./hp.yml"):
            os.remove("./hp.yml")

        self.config.dump_config(Path("./"), filename="hp.yml")

    def objective(self, trial):
        # config_file = "./hp.yml"
        # config = Config(Path(config_file))
        self.define_hparams(trial)
        if torch.cuda.is_available():
            metrics = start_hptuning(config_file=self.config_file)

        # fall back to CPU-only mode
        else:
            metrics = start_hptuning(config_file=self.config_file, gpu=-1)

        eval_metric = self.get_metrics(metrics, 'avg_loss',3)
        return eval_metric

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

    def run(self, config_file: Path, n_trials):
        LOGGER.info(f"Running {n_trials} experiments")
        self.config_file = config_file
        self.config = Config(config_file)

        self.study = self.create_study()
        
        LOGGER.info(f"Using: {self.study.sampler.__class__.__name__}")
        self.study.optimize(self.objective, n_trials=n_trials)
        
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()

    def get_best_params(self):
        trial = self.study.best_trial
        for key, value in trial.params.items():
            LOGGER.info("{}: {}".format(key, value))

    def save_best_params(self):
        trial = self.study.best_trial
        config_dict = self.config.as_dict()

        for key, value in trial.params.items():
            config_dict[key] = value
        print('Saved best config')
        self.config._cfg = config_dict

        if os.path.exists("./best_config.yml"):
            os.remove("./best_config.yml")

        self.config.dump_config(Path("./"), filename="best_config.yml")

    def create_study(self):
        if self.sampler_id == None or 0:
            return optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(), direction="minimize")
        elif self.sampler_id == 1:
            return optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), pruner=optuna.pruners.MedianPruner(), direction="minimize")
        else:
            return optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(), direction="minimize")