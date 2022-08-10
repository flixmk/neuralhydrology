import logging
import os
import optuna
from pathlib import Path
from neuralhydrology.utils.config import Config
import torch

from neuralhydrology.nh_run import start_hptuning

LOGGER = logging.getLogger(__name__)

class HpTuner():
    def __init__(self):
        self.method = 'bayesian'
        self.possible_methods = ['bayesian', 'genetic', 'tpe', 'pbt']
        self.tuner = None

        pass

    def select_method(self, method: str = None):
        if method is None or method == 'bayesian':
            self.method = 'bayesian'
            return BayesianTuner()
        elif method == 'genetic':
            self.method = 'genetic'
        elif method == 'tpe':
            self.method = 'tpe'
        elif method == 'pbt':
            self.method = 'pbt'
        else:
            LOGGER.error('The selected method is none of the offered optimization algorithms.\n'
                         'Try one of the following:\n'
                         '\'bayesian\' (default)\n'
                         '\'genetic\'\n'
                         '\'tpe\'(Tree-structured Parzen Esimators)\n'
                         '\'pbt\'(Population-based Training)\n')

    def set_params(self, params: dict):
        self.tuner.set_params(params)

    def run_tuning(self, config_file, method, params, n_runs):
        self.tuner = self.select_method(method)
        self.set_params(params)

        self.tuner.run(config_file, n_runs)

    def evaluate_runs(self):
        pass

    def save_best(self):
        self.tuner.get_best_params()
        self.tuner.save_best_params()


class Tuner():
    def __init__(self):
        pass

    def tune(self):
        pass

class BayesianTuner(Tuner):

    def __init__(self):
        super(BayesianTuner, self).__init__()
        self.config = None
        self.study = None
        self.param_names_range = None
        self.params = None
        self.predefined_ranges = {'hidden_size': (128, 512, 16),
                                  'output_dropout': (0.0,0.9,0.1),
                                  'initial_forget_bias': (0.0, 5.0, 0.5)}
        self.param_info = {'hidden_size': 'int',
                           'output_dropout': 'float',
                           'initial_forget_bias': 'float'}
        pass

    def set_params(self, params):
        self.param_names_range = params

    def define_hparams(self, trial):
        config_dict = self.config.as_dict()
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
            print(key, config_dict[key])

        self.config._cfg = config_dict

        if os.path.exists("./hp.yml"):
            os.remove("./hp.yml")

        self.config.dump_config(Path("./"), filename="hp.yml")

    def objective(self, trial):
        config_file = "./hp.yml"
        config = Config(Path(config_file))
        self.define_hparams(trial)
        if torch.cuda.is_available():
            val_loss = start_hptuning(config_file=self.config_file)

        # fall back to CPU-only mode
        else:
            val_loss = start_hptuning(config_file=self.config_file, gpu=-1)
        return val_loss

    def run(self, config_file: Path, n_trials):
        self.config_file = config_file
        self.config = Config(config_file)

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=n_trials, timeout=600)

    def get_best_params(self):
        trial = self.study.best_trial
        for key, value in trial.params.items():
            print("{}: {}".format(key, value))

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

