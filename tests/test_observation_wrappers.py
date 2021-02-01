from unittest import TestCase

import numpy as np
import gym

from duckietown_utils.wrappers.observation_wrappers import MotionBlurWrapper, ObservationBufferWrapper


class TestMotionBlurWrapper(TestCase):
    def test__angle_diff(self):
        self.assertEqual(MotionBlurWrapper._angle_diff(0, 0), 0.)
        self.assertEqual(MotionBlurWrapper._angle_diff(1., 1.), 0.)

        self.assertEqual(MotionBlurWrapper._angle_diff(0., np.pi / 2.), np.pi / 2.)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., np.pi), np.pi)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., 3 * np.pi / 2.), -np.pi / 2.)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., 2 * np.pi), 0.)

        self.assertEqual(MotionBlurWrapper._angle_diff(0., -np.pi / 2.), -np.pi / 2.)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., -np.pi), np.pi)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., -3 * np.pi / 2.), np.pi / 2.)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., -2 * np.pi), 0.)

        self.assertEqual(MotionBlurWrapper._angle_diff(0., 2 * np.pi + np.pi / 2.), np.pi / 2.)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., 2 * np.pi + np.pi), np.pi)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., 2 * np.pi + 3 * np.pi / 2.), -np.pi / 2.)
        self.assertEqual(MotionBlurWrapper._angle_diff(0., 2 * np.pi + 2 * np.pi), 0.)


class TestObservationBufferWrapper(TestCase):
    def setUp(self) -> None:
        self.obs_idx = 0
        self.dummy_env = gym.Env()
        self.dummy_env.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )
        self.obs_buffer = ObservationBufferWrapper(self.dummy_env, obs_buffer_depth=3)
        self.visualize_buffer()

    def test_observation(self):
        new_obs = np.zeros((84, 84, 3))
        self.obs_idx = 1
        buffered_obs = self.obs_buffer.observation(new_obs)
        np.testing.assert_array_equal(buffered_obs, np.zeros((84, 84, 9)))

    def visualize_buffer(self):
        for i in range(10):
            new_obs = np.ones((84, 84, 3)) * i
            buffered_obs = self.obs_buffer.observation(new_obs)
            print(buffered_obs[0, 0, :])
        self.obs_buffer.obs_buffer = None
        for i in range(10):
            new_obs = np.ones((84, 84, 3)) * i
            buffered_obs = self.obs_buffer.observation(new_obs)
            print(buffered_obs[0, 0, :])
