"""
Script for training lane-following agents (including collision avoidance).
Many properties of the training could be configured by modifying the config_updates dictionary at line 38-42.
For all available configuration options and their description, see config/config.yml and config/algo/ppo.yml
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"
###########################################################
# Imports
import os
from datetime import datetime
import logging
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.logger import CSVLogger, TBXLogger

from config.paths import ArtifactPaths
from config.config import load_config, print_config, dump_config, update_config, find_and_load_config_by_seed

from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.utils import seed
from duckietown_utils.rllib_callbacks import on_episode_start, on_episode_step, on_episode_end, on_train_result
from duckietown_utils.rllib_loggers import TensorboardImageLogger, WeightsAndBiasesLogger

logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################
# Load config
config = load_config('./config/config.yml', config_updates={"env_config": {"mode": "train"}})
# config = load_config('./config/config.yml', config_updates={})
# Set numpy and random seed
seed(1234)

###########################################################
# Set up experiment parameters
config_updates = {"seed": 0000,
                  "experiment_name": "Debug",
                  "env_config": {},
                  "rllib_config": {}
                  }
update_config(config, config_updates)

###########################################################
# Restore training
if config['restore_seed'] >= 0:
    pretrained_config, checkpoint_path = \
        find_and_load_config_by_seed(config['restore_seed'],
                                     preselected_experiment_idx=config['restore_experiment_idx'],
                                     preselected_checkpoint_idx=config['restore_checkpoint_idx'])
    logger.warning("Overwriting config from {}".format(checkpoint_path))
    config = pretrained_config
    update_config(config, config_updates)
else:
    checkpoint_path = None

###########################################################
# Print config
print_config(config)

###########################################################
# Setup paths
paths = ArtifactPaths(config['experiment_name'], config['seed'], algo_name=config['algo'])

###########################################################
# Code backup
os.system('cp -ar ./duckietown_utils {}/'.format(paths.code_backup_path))
os.system('cp -ar ./experiments {}/'.format(paths.code_backup_path))
os.system('cp -ar ./config {}/'.format(paths.code_backup_path))

###########################################################
# Set up env and training config
ray.init(**config["ray_init_config"])
register_env('Duckietown', launch_and_wrap_env)
config["rllib_config"].update({'env': 'Duckietown',
                               'callbacks': {'on_episode_start': on_episode_start,
                                             'on_episode_step': on_episode_step,
                                             'on_episode_end': on_episode_end,
                                             'on_train_result': on_train_result},
                               "env_config": config["env_config"],
                               })
dump_config(config, paths.experiment_base_path)

# Create an temporary PPO trainer to print the modell architecture (there should be a better way to do this)
# PPOTrainer(config=config["rllib_config"]).get_policy().model.base_model.summary()

###########################################################
# Run the training
tune.run(PPOTrainer,
         stop={'timesteps_total': config["timesteps_total"]},
         config=config["rllib_config"],
         local_dir="./artifacts",
         checkpoint_at_end=True,
         trial_name_creator=lambda trial: trial.trainable_name,  # for PPO this will make experiment dirs start with PPO_
         name=paths.experiment_folder,
         keep_checkpoints_num=1,
         checkpoint_score_attr="episode_reward_mean",
         checkpoint_freq=1,
         restore=checkpoint_path,
         loggers=[CSVLogger, TBXLogger, TensorboardImageLogger, WeightsAndBiasesLogger]
         )

