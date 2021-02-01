#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
Created based on the same script in gym_ducketown
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator
from duckietown_utils.wrappers.reward_wrappers import *
from duckietown_utils.wrappers.observation_wrappers import ClipImageWrapper, ResizeWrapper, MotionBlurWrapper, \
    RandomFrameRepeatingWrapper
from duckietown_utils.wrappers.simulator_mod_wrappers import ObstacleSpawningWrapper, ForwardObstacleSpawnnigWrapper
from duckietown_utils.wrappers.aido_wrapper import AIDOWrapper

from gym import envs
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if "Duckietown" in env_spec.id]
print(*env_ids, sep="\n")

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--map-name', default='loop_empty', help="Specify the map")
parser.add_argument('--distortion', action='store_true', help='Simulate lens distortion')
parser.add_argument('--draw-curve', action='store_true', help='Draw the lane following curve')
parser.add_argument('--domain-rand', action='store_true', help='Enable domain randomization')
parser.add_argument('--top-view', action='store_true',
                    help="View the simulation from a fixed bird's eye view, instead of the robot's view")
parser.add_argument('--spawn-vehicle-ahead', action='store_true',
                    help="Generate an obstacle vehicle a few tiles ahead of the controlled one")
parser.add_argument('--show-observations', action='store_true',
                    help='Show the cropped, downscaled observations, used as the policy input')
args = parser.parse_args()

if args.top_view:
    render_mode = 'top_down'
else:
    render_mode = 'human'

env = Simulator(
    seed=1234,
    map_name=args.map_name,
    domain_rand=args.domain_rand,
    dynamics_rand=args.domain_rand,
    camera_rand=args.domain_rand,
    distortion=args.distortion,
    accept_start_angle_deg=1,
    full_transparency=True,
    draw_curve=args.draw_curve,
    # user_tile_start=[2,1]
)
env = AIDOWrapper(env)
if args.show_observations:
    env = ClipImageWrapper(env, top_margin_divider=3)
    # env = ResizeWrapper(env, (84, 84))
# env = MotionBlurWrapper(env)
# env = RandomFrameRepeatingWrapper(env, {"frame_repeating": 0.33333})
# env = ObstacleSpawningWrapper(env, {'obstacles': {'duckie': {'density': 0.35,
#                                                              'static': True},
#                                                   'duckiebot': {'density': 0.25,
#                                                                 'static': True},
#                                                   'cone': {'density': 0.25,
#                                                                 'static': True},
#                                                   'barrier': {'density': 0.15,
#                                                            'static': True},
#                                                   },
#                                     'spawn_obstacles': True
#                                     }
if args.spawn_vehicle_ahead:
    env = ForwardObstacleSpawnnigWrapper(env, {'spawn_forward_obstacle': True})
# env = DtRewardWrapper(env)
# env = DtRewardClipperWrapper(env)
# env = DtRewardWrapperDistanceTravelled(env)
env = DtRewardPosAngle(env)
env = DtRewardVelocity(env)
env = DtRewardCollisionAvoidance(env)

env.reset()
env.reset()
env.render(render_mode)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)
def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    action = np.array([0.0, 0.0])
    if key_handler[key.UP]:
        action = np.array([1., 1.])
    if key_handler[key.DOWN]:
        action = np.array([-1., -1.])
    if key_handler[key.LEFT]:
        action = np.array([0, 1.])
    if key_handler[key.RIGHT]:
        action = np.array([1, 0.])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
    if key_handler[key.R]:
        action = np.array([0, 0])
        env.reset()

    obs, reward, done, info = env.step(action)
    if args.show_observations:
        # obs = cv2.resize(obs, (300, 300))
        cv2.imshow("Observation", cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)
        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render(render_mode)

    env.render(render_mode)

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
