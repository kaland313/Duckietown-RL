"""
Contains the definition and evaluation code of a PID controller for the Duckiebot, used as a baseline in our paper.
The controller relies on position and orientation error extracted from the simulation.
The PID controller is derived from the one used in gym-duckeitown.
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BaselinePIDAgent:
    def __init__(self):
        self.follow_dist = 0.15
        self.P = 0.5
        self.D = 5
        # self.trim = 0.0
        # self.radius = 0.0318
        # self.wheel_dist = WHEEL_DIST
        # self.robot_width = ROBOT_WIDTH
        # self.robot_length = ROBOT_LENGTH
        # self.gain = 2.
        # self.k = 27.0
        # self.limit = 1.0
        self.max_iterations = 1000
        self.prev_e = 0

    def step(self, simulator: Simulator):
        """
        Take a step, implemented as a PID controller
        """

        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = simulator.closest_curve_point(simulator.cur_pos, simulator.cur_angle)

        iterations = 0

        lookup_distance = self.follow_dist
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_tangent = simulator.closest_curve_point(follow_point, simulator.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - simulator.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        magic = (curve_tangent + point_vec) / np.linalg.norm(np.linalg.norm(point_vec))
        e = np.dot(self.get_right_vec(simulator.cur_angle), magic)
        de = e - self.prev_e
        self.prev_e = e
        steering = self.P * e + self.D * de
        return np.clip(np.array([1 + steering, 1 - steering]), 0., 1.)


    @staticmethod
    def get_right_vec(angle):
        x = math.sin(angle)
        z = math.cos(angle)
        return np.array([x, 0, z])


class DuckietownWorldEvaluatorBaselinePIDAgent(DuckietownWorldEvaluator):
    def _collect_trajectory(self, agent, i):
        episode_path = []
        episode_orientations = []
        episode_timestamps = []
        if self.map_name in self.start_poses.keys():
            self.env.unwrapped.user_tile_start = [0, 0]
            self.env.unwrapped.start_pose = self.start_poses[self.map_name][i]
        self.env.reset()
        done = False
        while not done:
            action = agent.step(self.env.unwrapped)
            obs, reward, done, info = self.env.step(action)
            cur_pos = correct_gym_duckietown_coordinates(self.env.unwrapped, self.env.unwrapped.cur_pos)
            episode_path.append(cur_pos)
            episode_orientations.append(np.array(self.env.unwrapped.cur_angle))
            episode_timestamps.append(info['Simulator']['timestamp'])
        self.env.unwrapped.start_pose = None
        self.user_tile_start = None
        return episode_path, episode_orientations, episode_timestamps


###########################################################
# Evaluate the pid agent
###########################################################
# Load config
config = load_config('./config/config.yml',
                     config_updates={"env_config": {"action_type": "default"}})
# Set numpy and random seed
seed(1234)
###########################################################
# Instantiate PID agent
pid_agent = BaselinePIDAgent()
###########################################################
# Plot trajectories and evaluate performance
evaluator = DuckietownWorldEvaluatorBaselinePIDAgent(config['env_config'])
evaluator.evaluate(pid_agent, "./pid_baseline_results")
