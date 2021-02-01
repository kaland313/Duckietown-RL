"""
Gym wrapper classes that transform the action space of Gym-Duckietown to alternative representations.
The original actions of Duckietown are the normalized wheel velocities of the robot.
These alternative action representations are
    - Discrete
    - Wheel velocity - Braking
    - Wheel velocity - Clipped (to [0, 1] interval)
    - Steering
    - etc.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import gym
import numpy as np
from gym import spaces


class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)
        # self.action_space = spaces.Box(low=0., high=1., shape=(3,))

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        # argmax_action = np.argmax(action)
        # sampled_action = np.random.sample([0, 1, 2, 3], 1, p=action)
        # Turn left
        if action == 0:
            vels = [0., 1.]
        #  Go forward
        elif action == 1:
            vels = [1., 1.]
        # Turn right
        elif action == 2:
            vels = [1., 0.]
        # # Stop
        # elif argmax_action == 3:
        #     vels = [0., 0.]
        else:
            assert False, "unknown action"
        return np.array(vels)


class LeftRightBraking2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(LeftRightBraking2WheelVelsWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        return np.clip(np.array([1., 1.]) - np.array(action), 0., 1.)


class LeftRightClipped2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(LeftRightClipped2WheelVelsWrapper, self).__init__(env)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        return np.clip(np.array(action), 0., 1.)


class Heading2WheelVelsWrapper(gym.ActionWrapper):
    def __init__(self, env, heading_type=None):
        super(Heading2WheelVelsWrapper, self).__init__(env)
        self.heading_type = heading_type
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        if self.heading_type == 'heading_trapz':
            straight_plateau_half_width = 0.3333  # equal interval for left, right turning and straight
            self.mul = 1. / (1. - straight_plateau_half_width)

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        # action = [-0.5 * action + 0.5, 0.5 * action + 0.5]
        if self.heading_type == 'heading_smooth':
            action = np.clip(np.array([1 + action ** 3, 1 - action ** 3]), 0., 1.)  # Full speed single value control
        elif self.heading_type == 'heading_trapz':
            action = np.clip(np.array([1 - action, 1 + action]) * self.mul, 0., 1.)
        elif self.heading_type == 'heading_sine':
            action = np.clip([1 - np.sin(action * np.pi), 1 + np.sin(action * np.pi)], 0., 1.)
        elif self.heading_type == 'heading_limited':
            action = np.clip(np.array([1 + action*0.666666, 1 - action*0.666666]), 0., 1.)
        else:
            action = np.clip(np.array([1 + action, 1 - action]), 0., 1.)  # Full speed single value control
        return action


class SteeringBraking2WheelVelsWrapper(gym.ActionWrapper):
    """
    Input: action vector
        action[0] - steering
        action[1] - braking
    Output: action vector:
        wheel velocities
    """
    def __init__(self, env, heading_type=None):
        super(SteeringBraking2WheelVelsWrapper, self).__init__(env)
        self.heading_type = heading_type
        self.action_space = spaces.Box(low=np.array([-1., 0.]), high=np.array([1., 1.]))

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        action = np.clip(np.array([1 + action[0], 1 - action[0]]), 0., 1.)  # Full speed single value control
        action *= np.clip(1. - action[1], 0., 1.)
        return action


class ActionSmoothingWrapper(gym.ActionWrapper):
    def __init__(self, env, ):
        super(ActionSmoothingWrapper, self).__init__(env)
        self.last_action = np.zeros(self.action_space.shape)
        self.new_action_ratio = 0.75

    def action(self, action):
        smooth_action = (1. - self.new_action_ratio) * self.last_action + self.new_action_ratio * action
        self.last_action = action
        return smooth_action

    def reset(self, **kwargs):
        self.last_action = np.zeros(self.action_space.shape)
        return self.env.reset(**kwargs)