"""
Takes a video that was recorded from a Duckiebot's perspective and outputs a video where salient objects are overlayed
over the original video, as if a specific reinforcement learning agent would have controlled the robot.
The input video can be specified at line 60
The agent is selected based on it's identifier (called seed), which is specified at line 40.
The method used to produce the salient object visualization is explained in:
Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car
https://arxiv.org/abs/1704.07911
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import time
import logging
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from config.config import *
from config.paths import ArtifactPaths
from duckietown_utils.env import launch_and_wrap_env, wrap_env, get_wrappers
from duckietown_utils.utils import seed
from duckietown_utils.rllib_callbacks import *
from duckietown_utils.salient_object_visualization import *

os.environ['CUDA_VISIBLE_DEVICES']=''

logger = logging.getLogger()
logger.setLevel(logging.INFO)

seed(1234)

###########################################################
# Load experiment
SEED = 3012
config, checkpoint_path = find_and_load_config_by_seed(SEED, preselected_experiment_idx=0, preselected_checkpoint_idx=1)
update_config(config, {'env_config': {'mode': 'inference',
                                      'spawn_forward_obstacle': False
                                      }})

###########################################################
# Set up env
ray.init(**config["ray_init_config"])
register_env('Duckietown', launch_and_wrap_env)

###########################################################
# Restore agent
trainer = PPOTrainer(config=config["rllib_config"])
trainer.restore(checkpoint_path)

print_config(trainer.config)

###########################################################
# Visualize
HIGH_RES_OUTPUT = True
model = tf.keras.models.clone_model(trainer.get_policy().model.base_model)  # type: tf.keras.Model

cap = cv2.VideoCapture('./docs/Real.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'FMP4')
fps = cap.get(cv2.CAP_PROP_FPS)
if HIGH_RES_OUTPUT:
    out = cv2.VideoWriter('salient_obj_video.mp4', cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 320))
else:
    out = cv2.VideoWriter('salient_obj_video.mp4', cv2.VideoWriter_fourcc(*"MJPG"), fps,
                          eval(config["env_config"]["resized_input_shape"]))

dummy_env = wrap_env(config["env_config"])
obs_wrappers, _, _ = get_wrappers(dummy_env)

action_dist_params_ = []

while cap.isOpened():
    retval, obs = cap.read()
    if retval:
        obs = cv2.resize(obs, (640, 480))
        obs = obs_high_res = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        for wrapper in obs_wrappers:
            obs = wrapper.observation(obs)

        if HIGH_RES_OUTPUT:
            displayed_obs = obs_wrappers[0].observation(obs_high_res)  # Clipping wrapper, shouldn't be hardcoded
            displayed_obs = cv2.resize(displayed_obs, (displayed_obs.shape[0], displayed_obs.shape[0]),
                                       interpolation=cv2.INTER_AREA)
            displayed_obs = (displayed_obs / 255.).astype(np.float32)
        else:
            displayed_obs = obs

        salient_map_mean, _ = nvidia_salient_map(model, obs)
        frame = display_salient_map2(salient_map_mean, displayed_obs, "Salient objects",
                                     frames_in_stack_to_be_displayed=[2], use_color_map=False)
        # salient_map_logstd, _ = nvidia_salient_map(model, obs, output_vector_idx=1)
        # display_salient_map(salient_map_logstd, obs, "Salient objects for action log stdev")
        frame = (frame * 255).astype(np.uint8)
        out.write(frame)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
