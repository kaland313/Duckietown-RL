"""
Gym wrapper classes that transform the observations from Gym-Duckietown to alternative representations.
- Clipping
- Resizing
- Normalization
- Stacking buffer
- Grayscale images
- Simple, convolution based motion blur simulation
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import gym
import cv2
from gym import spaces
import numpy as np
import logging
from gym_duckietown.simulator import CAMERA_FOV_Y

logger = logging.getLogger(__name__)

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160)):
        super(ResizeWrapper, self).__init__(env)
        if isinstance(shape, str):
            self.shape = eval(shape) + (self.observation_space.shape[2],)  # Depth is unchanged and can have any value
        else:
            self.shape = shape + (self.observation_space.shape[2],)  # Depth is unchanged and can have any value
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            self.shape,
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        resized = cv2.resize(observation, self.shape[:2][::-1], interpolation=cv2.INTER_AREA, )
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, 2)
        return resized


class ClipImageWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, top_margin_divider=3):
        super(ClipImageWrapper, self).__init__(env)
        img_height, img_width, depth = self.observation_space.shape
        top_margin = img_height // top_margin_divider
        img_height = img_height - top_margin
        # Region Of Interest
        # r = [margin_left, margin_top, width, height]
        self.roi = [0, top_margin, img_width, img_height]

        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        r = self.roi
        observation = observation[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        return observation

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ChannelsLast2ChannelsFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ChannelsLast2ChannelsFirstWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class ObservationBufferWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, obs_buffer_depth=3):
        super(ObservationBufferWrapper, self).__init__(env)
        obs_space_shape_list = list(self.observation_space.shape)

        # The last dimension, is used. For images, this should be the depth.
        # For vectors, the output is still a vector, just concatenated.
        self.buffer_axis = len(obs_space_shape_list) - 1
        obs_space_shape_list[self.buffer_axis] *= obs_buffer_depth
        self.observation_space.shape = tuple(obs_space_shape_list)

        if len(self.observation_space.shape) == 3:
            limit_low = self.observation_space.low[0, 0, 0]
            limit_high = self.observation_space.high[0, 0, 0]
        elif len(self.observation_space.shape) == 1:
            # Note this was implemented for vector like observation spaces (e.g. a VAE latent vector)
            limit_low = self.observation_space.low[0]
            limit_high = self.observation_space.high[0]
        else:
            assert False, "Only 1 or 3 dimentsional obs space supported!"

        self.observation_space = spaces.Box(
            limit_low,
            limit_high,
            tuple(obs_space_shape_list),
            dtype=self.observation_space.dtype)
        self.obs_buffer_depth = obs_buffer_depth
        self.obs_buffer = None

    def observation(self, obs):
        if self.obs_buffer_depth == 1:
            return obs
        if self.obs_buffer is None:
            self.obs_buffer = np.concatenate([obs for _ in range(self.obs_buffer_depth)], axis=self.buffer_axis)
        else:
            self.obs_buffer = np.concatenate((self.obs_buffer[..., (obs.shape[self.buffer_axis]):], obs), axis=self.buffer_axis)
        return self.obs_buffer

    def reset(self, **kwargs):
        self.obs_buffer = None
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

class RGB2GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(RGB2GrayscaleWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (self.observation_space.shape[0], self.observation_space.shape[1], 1),
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        # cv2.imshow("Camera", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("Camera2", gray)
        # cv2.waitKey(0)

        # Add an extra dimension, because conv lasers need an input as (batch, height, width, channels)
        gray = np.expand_dims(gray, 2)
        return gray


class MotionBlurWrapper(gym.ObservationWrapper):
    """
    Simulates motion blur separately for horizontal (left-right) rotational and forward movement.
    Both are simulated using a smoothing convolutional filter (with varying sized "horizontal" or "vertical" kernel).
    Forward motion should produce non-uniform blur (none at the center point of the horizon, and larger towards the
     edges), here it is simulated using the same filtering over the whole image (the size of the kernel depends on the
     robot velocity.

    According to gluPerspective (used by the Simulator) x is width, y is height.
    """
    def __init__(self, env):
        super(MotionBlurWrapper, self).__init__(env)
        self.camera_fov_height_angle = self.unwrapped.cam_fov_y / 180. * np.pi
        self.camera_fov_width_angle =\
            self.unwrapped.camera_width / float(self.unwrapped.camera_height) * self.camera_fov_height_angle
        self.camera_height_px = self.observation_space.shape[0]
        self.camera_width_px = self.observation_space.shape[1]
        self.blur_time = 0.05
        self.prev_angle = self.unwrapped.cur_angle
        self.simulate_rotational_blur = True
        self.simulate_forward_blur = False  # Vertical convolution doesn't work well for this

    def observation(self, observation):
        if self.simulate_rotational_blur:
            cur_angle = self.unwrapped.cur_angle
            angular_vel = self._angle_diff(self.prev_angle, cur_angle) * self.unwrapped.frame_rate * self.unwrapped.frame_skip
            self.prev_angle = cur_angle
            delta_angle = angular_vel * self.blur_time
            if abs(delta_angle) > 0:
                ksize = np.round(np.abs(delta_angle / self.camera_fov_width_angle * self.camera_width_px)).astype(int) + 1
                logger.debug("Rotational motion blur kernel size {}".format(ksize))
                kernel = np.zeros((ksize, ksize))
                kernel[ksize // 2, :] = 1. / ksize
                observation = cv2.filter2D(observation, -1, kernel)
        if self.simulate_forward_blur:
            # Empirical kernel size, proportional to the current speed over the max speed of the robot
            ksize = np.round(np.abs(self.camera_width_px / 30 * self.unwrapped.speed / self.unwrapped.robot_speed)).astype(int) + 1
            logger.debug("Forward motion blur kernel size {}".format(ksize))
            kernel = np.zeros((ksize, ksize))
            kernel[:, ksize // 2] = 1. / ksize
            observation = cv2.filter2D(observation, -1, kernel)
        return observation

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_angle = self.unwrapped.cur_angle
        return self.observation(observation)

    @staticmethod
    def _angle_diff(x, y):
        """
        Angle difference between two angles (orientations, directions). The smallest signed value is returned.
        The difference is measured from x to y (the difference is positive if y is larger)
        :param x, y: angles in radians
        :return: smallest angle difference between the two angles, should be in the range [-pi, pi]
        """
        diff = y-x
        remainder = diff % (2 * np.pi)
        quotient = diff // (2 * np.pi)
        if remainder > np.pi:
            diff -= 2 * np.pi
        diff -= quotient * 2 * np.pi
        return diff


class RandomFrameRepeatingWrapper(gym.ObservationWrapper):
    """
    Implements an evil domain randomisation option, replaces some observations with the previous.
    env_config["frame_repeating"]
    Values: 0...0.99 specify the probability of repeatingg
    """

    def __init__(self, env, env_config):
        super(RandomFrameRepeatingWrapper, self).__init__(env)
        self.repeat_config = np.clip(env_config["frame_repeating"], 0, 0.999)
        self.previous_frame = None

    def observation(self, observation):
        if self.previous_frame is None:
            self.previous_frame = observation
            return observation
        if np.random.random() < self.repeat_config:
            # New observation "not received" keeping the last one
            observation = self.previous_frame
        else:
            # New observation "received", store and don't change it
            self.previous_frame = observation
        return observation

    def reset(self, **kwargs):
        self.previous_frame = None
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

