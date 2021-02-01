"""
Utilities for managing settings using the config.yml file and other sources, such as updates to it in training scripts.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import os
import yaml
import glob
import logging
import numpy as np
from ray.tune.logger import pretty_print
from duckietown_utils.utils import recursive_dict_update

logger = logging.getLogger(__name__)


def load_config(path="./config/config.yml", update_algo_hparams_from_algo_conf_file=True, config_updates={}):
    """Loads configuation from config.yml
       WARNING: algo config can't be changed after being loaded (set update_algo_hparams_from_algo_conf_file=False)
    """
    with open(path) as f:
        config = yaml.load(f, yaml.FullLoader)

    if update_algo_hparams_from_algo_conf_file:
        if config_updates.get('algo') is not None:
            config["algo"] = config_updates['algo']
        config = load_algo_config(config)

    if config_updates != {}:
        update_config(config, config_updates)
    return config


def load_algo_config(config):
    """Loads algo specific config from algo config files, based on config['algo']"""
    algo = config['algo']
    if type(algo) == dict:  # most likely algo={'grid_search':[...]}
        algo = 'general'
    if algo not in config["algo_config_files"].keys():
        algo = 'general'
    algo_config_file = config["algo_config_files"][algo]
    with open(algo_config_file) as f:
        algo_config = yaml.load(f, yaml.FullLoader)
    config.update(algo_config)
    return config


def dump_config(config, path):
    file_path = os.path.join(path, "config_dump_{:04d}.yml".format(config["seed"]))
    with open(file_path, "w") as config_dump:
        yaml.dump(config, config_dump, yaml.Dumper)


def print_config(config: dict):
    logger.info("=== Config ===================================")
    logger.info(pretty_print(config))


def update_config(config: dict, config_updates: dict):
    logger.warning("Updating default config values by: \n {}".format(pretty_print(config_updates)))
    recursive_dict_update(config, config_updates)

    # If the seed and experiment_name are changed their copies in env config should be updated as well
    # (In the config.yml file this is done by anchors and alias indicators
    config['env_config'].update({'seed': config['seed'],
                                 'experiment_name': config['experiment_name']})

    if 'mode' in config['env_config'].keys():
        if config['env_config']['mode'] == 'debug':
            logger.warning(
                "Env_config.mode is 'debug', some hyperparameters will be overwritten by: \n {}".format(
                    pretty_print(config["debug_hparams"])))
            config["rllib_config"].update(config["debug_hparams"]["rllib_config"])
            config["ray_init_config"].update(config["debug_hparams"]["ray_init_config"])

        default_config = load_config(update_algo_hparams_from_algo_conf_file=False)
        if 'inference_hparams' not in config.keys():
            config['inference_hparams'] = default_config['inference_hparams']
        elif 'explore' not in config["inference_hparams"]["rllib_config"]:
            config["inference_hparams"]["rllib_config"]['explore'] = \
                default_config["inference_hparams"]["rllib_config"]['explore']
            # Setting explore to what is set in the default config (false) is important, because in many older trainings
            # this key is missing, in which case it is treated as true by rllib.

        if config['env_config']['mode'] == 'inference':
            logger.warning(
                "Env_config.mode is 'inference', some hyperparameters will be overwritten by: \n {}".format(
                    pretty_print(config["inference_hparams"])))
            config["rllib_config"].update(config["inference_hparams"]["rllib_config"])
            config["ray_init_config"].update(config["inference_hparams"]["ray_init_config"])

        assert config['env_config']['mode'] in ['train', 'inference', 'debug']

    # For loaded config dups the env config is replicated in rllib_config
    if 'env_config' in config['rllib_config'].keys():
        config['rllib_config']['env_config'].update(config['env_config'])


def find_and_load_config_by_seed(seed, artifact_root="./artifacts",
                                 preselected_experiment_idx=None, preselected_checkpoint_idx=None):
    logger.warning("Found paths with seed {}:".format(str(seed)))
    config_dump_path = _find_and_select_experiment(artifact_root + '/**/config_dump_{:04d}.yml'.format(seed),
                                                   preselected_experiment_idx)

    # Multiple checkpoints might be saved under the same experiment folder
    logger.warning("Found checkpoints in {}:".format(os.path.dirname(config_dump_path)))
    # *[0-9] makes sure that the last character is a number --> the .tune_metadata files are excluded
    checkpoint_path = _find_and_select_experiment(
        os.path.dirname(config_dump_path) + '/**/checkpoint-*[0-9]'.format(seed),
        preselected_checkpoint_idx)

    loaded_config = load_config(config_dump_path, update_algo_hparams_from_algo_conf_file=False)
    logger.warning("Config loaded from {}".format(config_dump_path))
    logger.warning("Model checkpoint loaded from {}".format(checkpoint_path))
    return loaded_config, checkpoint_path


def _find_and_select_experiment(search_string, preselect_index=None):
    paths = glob.glob(search_string, recursive=True)
    paths.sort()
    for i, path in enumerate(paths):
        logger.warning("{:d}: {}".format(i, path))

    number_of_experiments = len(paths)
    if number_of_experiments <= 0:
        assert False, "No artifacts found with with pattern {}".format(search_string)

    if number_of_experiments > 1:  # more than one experiment was found with this seed
        if preselect_index is None:
            logger.warning("Enter experiment number: ")
            experiment_num = int(input())
        else:
            experiment_num = preselect_index
        experiment_num = np.clip(experiment_num, 0, number_of_experiments - 1)
    else:
        experiment_num = 0

    return paths[experiment_num]
