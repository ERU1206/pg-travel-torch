import torch
import numpy as np
import random

def set_random_seed(seed:int = 777):
    """
    set random seed
    :param seed:
    :return:
    """
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)