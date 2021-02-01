"""
Custom utilities to log training metrics from RLlib to Tensorboad and Weights & Biases.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"


import os
# from hyperdash import Experiment
from tensorboardX import SummaryWriter
import wandb

from config.paths import ArtifactPaths


def _flatten_all_levels(recursive_dict):
    while True:
        flattened_dict = _flatten_dict_of_dicts(recursive_dict)
        if recursive_dict == flattened_dict:
            break
        else:
            recursive_dict = flattened_dict
    return flattened_dict


def _flatten_dict_of_dicts(dict_of_dicts):
    flattened_dict = {}
    for key, val in dict_of_dicts.items():
        if isinstance(val, dict):
            for inner_key, inner_val in val.items():
                flattened_dict[key + '/' + inner_key] = inner_val
        else:
            flattened_dict[key] = val
    return flattened_dict


class CompoundExperimentTracker:
    def __init__(self, args, paths: ArtifactPaths, enable_tensorboard=True, enable_wand=True,
                 wandb_project="duckietown-rl"):
        self.enable_tensorboard = enable_tensorboard
        self.enable_wand = enable_wand
        if self.enable_tensorboard:
            self.tensorboard = SummaryWriter(logdir=paths.tensorboard_path)
        if self.enable_wand:
            self.wandb_run = wandb.init(project=wandb_project, name=args.experiment_name)
        self.global_step = 1
        self.log_args(args)
        # self.hyperdash = Experiment(str(args.algo) + " - " + args.experiment_name)
        # for arg in vars(args):
        #     hyperdash_logger.param(arg, getattr(args, arg))

    def log_args(self, args):
        if self.enable_wand:
            wandb.config.update(args)

    def add_scalar_dict(self, recursive_dict, global_step=None):
        flattened_dict = _flatten_all_levels(recursive_dict)
        for key, val in flattened_dict.items():
            self.add_scalar(key, val, global_step)

    def add_scalar(self, tag, scalar_value, global_step=None):
        if global_step is None:
            global_step = self.global_step
            self.global_step +=1
        else:
            self.global_step = global_step
        if self.enable_tensorboard:
            self.tensorboard.add_scalar(tag, scalar_value, global_step=global_step)
        if self.enable_wand:
            wandb.log({tag: scalar_value}, step=global_step)

    def add_histogram(self, tag, values, global_step=None, bins='auto'):
        if global_step is None:
            global_step = self.global_step
            self.global_step +=1
        else:
            self.global_step = global_step
        if self.enable_tensorboard:
            self.tensorboard.add_histogram(tag, values, bins=bins, global_step=global_step)
        if self.enable_wand:
            # wandb.log({tag: wandb.Histogram(values, num_bins=bins)})
            wandb.log({tag: wandb.Histogram(values)}, step=global_step)

    def add_figure(self, tag, figure, global_step=None):
        if self.enable_tensorboard:
            self.tensorboard.add_figure(tag, figure, global_step=global_step)
        if self.enable_wand:
            wandb.log({tag: wandb.Image(figure)}, step=global_step)

    def add_image(self, tag, img_tensor, global_step=None, dataformats='CHW'):
        if self.enable_tensorboard:
            self.add_image(tag, img_tensor, global_step=global_step, dataformats=dataformats)
        if self.enable_wand:
            wandb.log({tag: wandb.Image(img_tensor)}, step=global_step)


if __name__ == "__main__":
    # execute only if run as a script to test some functionality
    recursive_dict = {'A': {'a': {'1': "A/a/1",
                                  '2': "A/a/2"},
                            'b': "A/b",
                            'c': "A/c"},
                      'B': {'a': {'1': "B/a/1",
                                  '2': "B/a/2"}}}
    print(_flatten_all_levels(recursive_dict))

