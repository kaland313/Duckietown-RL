"""
Record and evaluate how a human can control a simulated Duckiebot.
A simulator windows should appear soon after starting the script, where you can control the robot using the arrow keys.
After the termination of an episode the recorded trajectory will be analysed (you might need to wait a few seconds for
this), then the next episode will start.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import numpy as np
import logging
import math
from gym_duckietown.simulator import Simulator, ROBOT_LENGTH, ROBOT_WIDTH, WHEEL_DIST
from duckietown_utils.duckietown_world_evaluator import DuckietownWorldEvaluator, DEFAULT_EVALUATION_MAP, myTestMapA
from duckietown_utils.trajectory_plot import correct_gym_duckietown_coordinates
from duckietown_utils.utils import seed
from config.config import load_config
import pyglet
from pyglet.window import key

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DuckietownWorldEvaluatorHumanBaseline(DuckietownWorldEvaluator):
    def __init__(self, env_config, eval_lenght_sec=15, eval_map=DEFAULT_EVALUATION_MAP):
        super(DuckietownWorldEvaluatorHumanBaseline, self).__init__(env_config, eval_lenght_sec, eval_map)
        self.episode_path = []
        self.episode_orientations = []
        self.episode_timestamps = []
        self.env.reset()
        self.env.render()
        # Register a keyboard handler
        self.key_handler = key.KeyStateHandler()
        self.env.unwrapped.window.push_handlers(self.key_handler)
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        self.joystick = None
        joysticks = pyglet.input.get_joysticks()
        if joysticks:
            self.joystick = joysticks[0]
            self.joystick.open()

    def _collect_trajectory(self, agent, i):
        self.episode_path = []
        self.episode_orientations = []
        self.episode_timestamps = []
        if self.map_name in self.start_poses.keys():
            self.env.unwrapped.user_tile_start = [0, 0]
            self.env.unwrapped.start_pose = self.start_poses[self.map_name][i % len(self.start_poses[self.map_name])]
        self.env.reset()
        self.env.render()
        # Enter main event loop
        pyglet.app.run()

        return self.episode_path, self.episode_orientations, self.episode_timestamps

    def update(self, dt):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """
        action = np.array([0.0, 0.0])
        if self.key_handler[key.UP]:
            action = np.array([1., 1.])
        if self.key_handler[key.DOWN]:
            action = np.array([-1., -1.])
        if self.key_handler[key.LEFT]:
            action = np.array([0, 1.])
        if self.key_handler[key.RIGHT]:
            action = np.array([1, 0.])
        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        if self.joystick is not None:
            # action = np.array([self.joystick.z, self.joystick.rz])
            steering = self.joystick.rx
            action = np.clip(np.array([1 + steering, 1 - steering]), 0., 1.) * np.clip(self.joystick.z, 0, 1)

        obs, reward, done, info = self.env.step(action)
        cur_pos = correct_gym_duckietown_coordinates(self.env.unwrapped, self.env.unwrapped.cur_pos)
        self.episode_path.append(cur_pos)
        self.episode_orientations.append(np.array(self.env.unwrapped.cur_angle))
        self.episode_timestamps.append(info['Simulator']['timestamp'])
        if done:
            pyglet.app.exit()

        self.env.render()


###########################################################
# Evaluate the human baseline
###########################################################
# Load config
config = load_config('./config/config.yml',
                     config_updates={"env_config": {"action_type": "default",
                                                    "reward_function": 'default',
                                                    "resized_input_shape": '(480, 640)',
                                                    "crop_image_top": False,
                                                    "frame_stacking": False,
                                                    "distortion": False
                                                    }})
# Set numpy and random seed
seed(1234)

###########################################################
# Plot trajectories and evaluate performance
evaluator = DuckietownWorldEvaluatorHumanBaseline(config['env_config'])
evaluator.evaluate(None, "./human_baseline_results", episodes=15)
