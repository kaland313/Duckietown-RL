"""
Script to evaluate our agents in many ways.
For the possible evaluation method options take a look at the arguments below.
To select a trained agent use the --seed-model-id argument (e.g. --seed-model-id 3045)
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import time
from tqdm import tqdm
import os
import logging
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import cv2
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from gym.wrappers import Monitor

from config.config import find_and_load_config_by_seed, update_config, print_config
from config.paths import ArtifactPaths
from duckietown_utils.env import launch_and_wrap_env, get_wrappers
from duckietown_utils.utils import seed
from duckietown_utils.rllib_callbacks import *
from duckietown_utils.duckietown_world_evaluator import DuckietownWorldEvaluator, DEFAULT_EVALUATION_MAP, myTestMapA
from duckietown_utils.trajectory_plot import correct_gym_duckietown_coordinates, plot_trajectories
from duckietown_utils.salient_object_visualization import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES']=''

###########################################################
# Read and process command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed-model-id', default=3045, type=int,
                    help='Unique experiment identifier, referred to as seed (incorrectly)'
                         'A 4 digit number. Selected models: 3012, 3045, 3090, 3092')
parser.add_argument('--analyse-trajectories', action='store_true',
                    help='Calculate metrics and create trajectory plots.')
parser.add_argument('--visualize-salient-obj', action='store_true',
                    help='Visualize salient object, while running the policy in clsed loop simulation')
parser.add_argument('--visualize-dot-trajectories', action='store_true',
                    help='Visualize trajectories of many episodes as dotted lines')
parser.add_argument('--reward-plots', action='store_true',
                    help='Simulate closed loop behaviour and show time-plots of the reward, '
                         'distance between vehicles, etc.')
parser.add_argument('--map-name', default=DEFAULT_EVALUATION_MAP, help="Specify the map")
parser.add_argument('--domain-rand', action='store_true', help='Enable domain randomization')
parser.add_argument('--top-view', action='store_true',
                    help="View the simulation from a fixed bird's eye view, instead of the robot's view")
parser.add_argument('--results-path', default='default', type=str,
                    help='Analysis results are saved to this folder. If \'default\' is given, results are saved to '
                         'the path of the loaded model.')
args = parser.parse_args()

if args.top_view:
    render_mode = 'top_down'
else:
    render_mode = 'human'

test_map = args.map_name

seed(1234)

###########################################################
# Load experiment
SEED = args.seed_model_id  # Experiment ID
config, checkpoint_path = find_and_load_config_by_seed(SEED, preselected_experiment_idx=0, preselected_checkpoint_idx=0)
update_config(config, {'env_config': {'mode': 'inference',
                                      'training_map': test_map,  # This controls what is used in the demo part
                                      'domain_rand': False
                                      }})

# Set up env
ray.init(**config["ray_init_config"])
register_env('Duckietown', launch_and_wrap_env)

###########################################################
# Restore agent
trainer = PPOTrainer(config=config["rllib_config"])
trainer.restore(checkpoint_path)

print_config(trainer.config)

###########################################################
###########################################################
# Simple demonstration of closed loop performance
if not (args.analyse_trajectories or args.visualize_salient_obj or args.reward_plots or args.visualize_dot_trajectories):
    # env = Monitor(env, "gym_monitor_results", write_upon_reset=True, force=True)
    env = launch_and_wrap_env(config["env_config"])
    for i in range(5):
        obs = env.reset()
        env.render(render_mode)
        done = False
        while not done:
            action = trainer.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            cv2.imshow("Observation", cv2.cvtColor(cv2.resize(obs[..., -3:].astype('float32'), (300, 300)), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            orig_distorion = env.unwrapped.distortion
            env.unwrapped.distortion = False
            env.render(render_mode)
            env.unwrapped.distortion = orig_distorion
            if env.unwrapped.frame_skip > 1:
                time.sleep(env.unwrapped.delta_time * env.unwrapped.frame_skip)

###########################################################
# Plot trajectories and evaluate performance
if args.analyse_trajectories:
    config['env_config']['spawn_forward_obstacle'] = False  # Works if True, but visualisations won't be incorrect
    evaluator = DuckietownWorldEvaluator(config['env_config'], eval_lenght_sec=15, eval_map=test_map)
    if args.results_path == 'default':
        results_path = os.path.split(checkpoint_path)[0]
    else:
        results_path = args.results_path
    evaluator.evaluate(trainer, results_path)

###########################################################
# Visualize salient map
if args.visualize_salient_obj:
    HIGH_RES_OUTPUT = True
    if HIGH_RES_OUTPUT:
        out = cv2.VideoWriter('salient_obj_video.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 320))
    else:
        out = cv2.VideoWriter('salient_obj_video.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 30,
                              eval(config["env_config"]["resized_input_shape"]))

    model = tf.keras.models.clone_model(trainer.get_policy().model.base_model)  # type: tf.keras.Model
    env = launch_and_wrap_env(config["env_config"])
    obs_wrappers, _, _ = get_wrappers(env)
    env.reset()
    obs = env.reset()
    done = False
    while not done:
        salient_map_mean, action_dist_params = nvidia_salient_map(model, obs)
        if HIGH_RES_OUTPUT:
            render = env.render_obs()
            displayed_obs = obs_wrappers[0].observation(render)  # Clipping wrapper, shouldn't be hardcoded
            displayed_obs = cv2.resize(displayed_obs, (displayed_obs.shape[0], displayed_obs.shape[0]),
                                       interpolation=cv2.INTER_AREA)
            displayed_obs = (displayed_obs / 255.).astype(np.float32)
        else:
            displayed_obs = obs
        frame = display_salient_map2(salient_map_mean, displayed_obs, "Salient objects", frames_in_stack_to_be_displayed=[2],
                             use_color_map=False)
        action = trainer.compute_action(obs, explore=False)
        obs, reward, done, info = env.step(action)
        out.write((frame * 255).astype(np.uint8))

    out.release()

###########################################################
# Dotted trajectory plots
if args.visualize_dot_trajectories:
    config['env_config']['training_map'] = 'multimap1'
    trajectories = []
    for i in tqdm(range(10)):
        env = launch_and_wrap_env(config["env_config"], i)
        ego_robot_pos = []
        obs = env.reset()
        done = False
        while not done:
            action = trainer.compute_action(obs, explore=False)
            obs, reward, done, info = env.step(action)
            ego_robot_pos.append(correct_gym_duckietown_coordinates(env.unwrapped, env.unwrapped.cur_pos))
        trajectories.append(ego_robot_pos)
        env.close()

    plot_trajectories(trajectories, show_plot=True)
    plot_trajectories(trajectories, show_plot=True, unify_start_tile=False)

###########################################################
# Detailed demonstration of closed loop performance
if args.reward_plots:
    from gym_duckietown.objects import DuckiebotObj
    for i in range(1):
        env = launch_and_wrap_env(config["env_config"], i)
        prox_pen = []
        coll_reward = []
        vel_reward = []
        angl_reward = []
        ego_robot_pos = []
        npc_robot_pos = []
        timestamps = []
        # Specify start position
        # env.unwrapped.user_tile_start = [0, 0]
        # env.unwrapped.start_pose = [[0.585/4, 0, 0.585], -np.pi/2]
        obs = env.reset()
        env.render('top_down')
        done = False
        step = 0
        while not done:
            t0 = time.time()
            action = trainer.compute_action(obs, explore=False)
            # print(action[0])
            t1 = time.time()
            obs, reward, done, info = env.step(action)
            prox_pen.append(info['Simulator']['proximity_penalty'])
            coll_reward.append(info.get('custom_rewards', {}).get('collision_avoidance', 0))
            vel_reward.append(info.get('custom_rewards', {}).get('velocity', 0))
            angl_reward.append(info.get('custom_rewards', {}).get('orientation', 0))
            timestamps.append(info['Simulator']['timestamp'])
            ego_robot_pos.append(correct_gym_duckietown_coordinates(env.unwrapped, env.unwrapped.cur_pos))
            for npc in env.unwrapped.objects:
                if isinstance(npc, DuckiebotObj):
                    npc_robot_pos.append(correct_gym_duckietown_coordinates(env.unwrapped, npc.pos))
            t2 = time.time()
            env.unwrapped.distortion = False
            env.render(render_mode)
            #if step in np.array(range(30))*30:
                # cv2.imwrite("./LFV/TopView{:4.3f}.png".format(info['Simulator']['timestamp']-0.0333333333), cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR))
            step +=1
            env.unwrapped.distortion = True
            t3 = time.time()
            print("Inference time {:4.3f}ms | Env step time {:4.3f}ms | Render time {:4.3f}ms".format((t1-t0)*1000,
                  (t2-t1)*1000, (t3-t2)*100))

        matplotlib.use('TkAgg')
        plt.subplot(211)
        plt.plot(np.array(prox_pen))
        plt.plot(np.array(coll_reward))
        plt.legend(["Proximity penalty", "Collision Avoidance Reward"])
        plt.grid('on')
        plt.subplot(212)
        plt.plot(np.array(vel_reward))
        plt.plot(np.array(angl_reward))
        plt.plot(np.array(coll_reward)+np.array(vel_reward)+np.array(angl_reward))
        plt.legend(["Velocity", "Orientation", "Sum"])
        plt.ylabel("Reward components")
        plt.grid('on')
        plt.show()

        if len(npc_robot_pos) > 0:
            ROBOT_LENGTH = 0.18
            plt.plot(np.array(timestamps), np.linalg.norm(np.array(npc_robot_pos)-np.array(ego_robot_pos), axis=1)-ROBOT_LENGTH)
            plt.ylabel("Distance between robot centers[m]")
            plt.grid('on')
            plt.xlabel("Time [s]")
            plt.show()

        plt.plot(np.array(timestamps[1:]), env.unwrapped.frame_rate *
                  np.linalg.norm(np.array(ego_robot_pos)[1:] - np.array(ego_robot_pos)[:-1], axis=1))
        if len(npc_robot_pos) > 0:
            plt.plot(np.array(timestamps[1:]), env.unwrapped.frame_rate *
                     np.linalg.norm(np.array(npc_robot_pos)[1:] - np.array(npc_robot_pos)[:-1], axis=1))
            plt.legend(["Controlled robot", "Obstacle robot"])
        plt.ylabel("Robot speed [m/s]")
        plt.grid('on')
        plt.xlabel("Time [s]")
        plt.show()

