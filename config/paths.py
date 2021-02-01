"""
Module that manages the paths for saving and loading training artifacts based on a single identifier (called seed).
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import os
from datetime import datetime
import glob
import numpy as np
import string
import logging

logger = logging.getLogger(__name__)

class ArtifactPaths:

    def __init__(self, experiment_name="training", seed=0, experiment_base_path=None, artifact_root="./artifacts",
                 algo_name=""):
        if experiment_base_path is None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.experiment_folder = os.path.join("{}_{:04d}".format(experiment_name, seed), current_time)
            self.experiment_base_path = os.path.join(artifact_root, self.experiment_folder)
            if not os.path.exists(self.experiment_base_path):
                os.makedirs(self.experiment_base_path)
            make_folder = True
        else:
            self.experiment_base_path = experiment_base_path
            make_folder = False

        # self.ray_results_path = self._check_create_folder("ray_results", make_folder)
        # self.models_path = self._check_create_folder("models", make_folder)
        # self.tensorboard_path = self._check_create_folder("tensorboard", make_folder)
        # self.video_path = self._check_create_folder("videos", make_folder)
        self.code_backup_path = self._check_create_folder("source_code", make_folder)
        # self.dt_word_eval_path = self._check_create_folder("dt_world_eval", make_folder)

        # self.json_path = os.path.join(self.experiment_base_path, "args_" + str(seed) + ".json")


        self.model_file_name = "{}_{}".format(str(algo_name).upper(), str(seed))

        print("==============================================")
        print("Artifacts paths: ")
        print(self.experiment_base_path)
        # print(self.ray_results_path)
        # print(self.models_path)
        # print(self.tensorboard_path)
        # print(self.video_path)
        print(self.code_backup_path)

    def _check_create_folder(self, folder, make_path=True):
        folder_path = os.path.join(self.experiment_base_path, folder)
        if not os.path.exists(folder_path):
            if make_path:
                os.makedirs(folder_path)
            else:
                logger.warning("Folder not found:".format(folder))
        return folder_path
