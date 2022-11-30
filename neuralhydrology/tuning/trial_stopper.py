from typing import Dict, Optional
from collections import defaultdict, deque
import numpy as np

from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper
from ray.tune.stopper import CombinedStopper,MaximumIterationStopper, TrialPlateauStopper


class Trial_Stopper(Stopper):
    def __init__():
        pass
    
class Trial_Metric_Stopper(Stopper):