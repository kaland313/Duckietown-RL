"""
Misc utils.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"
import random
import numpy as np


def seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def recursive_dict_update(target_dict, update_dict):
    for key, val in update_dict.items():
        if type(val) is dict and type(target_dict.get(key)) is dict:
            target_dict[key] = recursive_dict_update(target_dict.get(key, {}), val)
        else:
            target_dict[key] = val
    return target_dict



