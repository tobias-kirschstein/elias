import random

import numpy as np


def make_deterministic(seed: int = 0):
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    random.seed(seed)
    np.random.seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
