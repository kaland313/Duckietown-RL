"""
RLlib callbacks used to log custrom metrics and a hack to save trajectory data.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
import numpy as np
import ray.rllib as rllib
from ray.tune.logger import pretty_print
from ray.tune.result import TRAINING_ITERATION, TIMESTEPS_TOTAL

logger = logging.getLogger(__name__)

from .trajectory_plot import correct_gym_duckietown_coordinates
from .env import resolve_multimap_name
from .wrappers.simulator_mod_wrappers import ObstacleSpawningWrapper

# The actions returned by the rllib agent are not clipped to be within action space bounds, thus they can be arbitrarily
# large. To make the histograms correct the actions should be clipped to the action space bound.
# ACTION_HISTOGRAM_LIMITS should hold the upper and lower bound of the action space.
ACTION_HISTOGRAM_LIMITS = [-1., 1.]

def on_episode_start(info):
    """
    If adding new histograms, don't forget to edit on_train_result below (to prevent data accumulation over iterations).
    :param info:
    :return:
    """
    # info-keys: 'env', 'policy', 'episode'
    episode = info['episode']  # type: rllib.evaluation.episode.MultiAgentEpisode

    episode.user_data['robot_speed'] = []
    episode.user_data['robot_cur_pos'] = []
    episode.user_data['deviation_centerline'] = []
    episode.user_data['deviation_heading'] = []
    episode.user_data['distance_travelled'] = []
    episode.user_data['distance_travelled_any'] = []
    episode.user_data['proximity_penalty'] = []
    episode.user_data['collision_risk_step_cnt'] = 0
    episode.user_data['reward_orientation'] = []
    episode.user_data['reward_velocity'] = []
    episode.user_data['reward_collision_avoidance'] = []
    # Custom histogram data
    # episode.hist_data['action_prob'] = []
    episode.hist_data['sampled_actions'] = []
    episode.hist_data['_robot_coordinates'] = []

def on_episode_step(info):
    episode = info['episode']  # type: rllib.evaluation.episode.MultiAgentEpisode
    # info-keys: 'env', 'episode'
    episode.hist_data['sampled_actions'].append(np.clip(episode.last_action_for(),
                                                        ACTION_HISTOGRAM_LIMITS[0], ACTION_HISTOGRAM_LIMITS[1]))
    env_info = episode.last_info_for()

    # {'Simulator': {'action': [array([0.96753883], dtype=float32), array([1.], dtype=float32)],
    #                'lane_position': {'dist': -0.09179686463148151,
    #                                  'dot_dir': 0.9997813004067312,
    #                                  'angle_deg': 1.1983109648377053,
    #                                  'angle_rad': 0.020914471799167954},
    #                'robot_speed': 0.0,
    #                'proximity_penalty': 0,
    #                'cur_pos': [3.859709301028824, 0.0, 4.362296864631481],
    #                'cur_angle': 3.1206781817906233,
    #                'wheel_velocities': [array([1.1610466], dtype=float32), array([1.2], dtype=float32)],
    #                'timestamp': 0.03333333333333333,
    #                'tile_coords': [6, 7],
    #                'msg': ''}}
    if env_info is not None:
        episode.user_data['robot_speed'].append(env_info['Simulator']['robot_speed'])
        episode.user_data['proximity_penalty'].append(env_info['Simulator']['proximity_penalty'])
        if env_info['Simulator']['proximity_penalty'] < 0.:
            episode.user_data['collision_risk_step_cnt'] += 1
        episode.user_data['reward_orientation'].append(env_info.get('custom_rewards', {}).get('orientation', 0.))
        episode.user_data['reward_velocity'].append(env_info.get('custom_rewards', {}).get('velocity', 0.))
        episode.user_data['reward_collision_avoidance'].append(env_info.get('custom_rewards', {}).get('collision_avoidance', 0.))
        # If the robot is "not in a lane", the lane position key is not added to the simulator info dictionary
        # see gym_duckietown.simulator.Simulator.get_agent_info()  (line 1318)
        if 'lane_position' in env_info['Simulator'].keys():
            episode.user_data['deviation_centerline'].append(abs(env_info['Simulator']['lane_position']['dist']))
            episode.user_data['deviation_heading'].append(abs(env_info['Simulator']['lane_position']['angle_deg']))

        cur_pos = env_info['Simulator']['cur_pos']
        sim = info['env'].get_unwrapped()[0].unwrapped
        corrected_cur_pos = correct_gym_duckietown_coordinates(sim, cur_pos)
        episode.user_data['robot_cur_pos'].append(corrected_cur_pos)

        dist_travelled = 0.  # Distance traveled in the correct right side lane
        dist_travelled_any = 0.  # Distance traveled anywhere on the road
        if 'lane_position' in env_info['Simulator'].keys():
            if len(episode.user_data['robot_cur_pos']) > 1:
                dist_travelled_any = np.linalg.norm(episode.user_data['robot_cur_pos'][-1] -
                                                episode.user_data['robot_cur_pos'][-2], ord=2)
                if env_info['Simulator']['lane_position']['dist'] > -0.1:
                    # driving in the correct lane
                    dist_travelled = dist_travelled_any
        episode.user_data['distance_travelled'].append(dist_travelled)
        episode.user_data['distance_travelled_any'].append(dist_travelled_any)

    # try:
    #     policy_info = episode.last_pi_info_for()
    #     episode.hist_data['action_prob'].append(policy_info['action_prob'])
    # except KeyError as err:
    #     logger.warning("KeyError {}".format(err))


def on_episode_end(info):
    # info-keys: 'env', 'policy', 'episode'
    episode = info['episode']  # type: rllib.evaluation.episode.MultiAgentEpisode

    episode.custom_metrics['mean_robot_speed'] = np.mean(episode.user_data['robot_speed'])
    episode.custom_metrics['deviation_centerline'] = np.mean(episode.user_data['deviation_centerline'])
    episode.custom_metrics['deviation_heading'] = np.mean(episode.user_data['deviation_heading'])
    episode.custom_metrics['distance_travelled'] = np.sum(episode.user_data['distance_travelled'])
    episode.custom_metrics['distance_travelled_any'] = np.sum(episode.user_data['distance_travelled_any'])
    episode.custom_metrics['proximity_penalty'] = np.sum(episode.user_data['proximity_penalty'])
    episode.custom_metrics['collision_risk_step_cnt'] = episode.user_data['collision_risk_step_cnt']
    episode.custom_metrics['reward_orientation'] = np.sum(episode.user_data['reward_orientation'])
    episode.custom_metrics['reward_velocity'] = np.sum(episode.user_data['reward_velocity'])
    episode.custom_metrics['reward_collision_avoidance'] = np.sum(episode.user_data['reward_collision_avoidance'])

    # Robot coordinate data is not intended to be displayed on histograms (it's not even in the correct format for it)
    # Robot coordinates are logged as histogram data because I couldn't find a better way to pass it to the loggers
    # to produce the trajectory plots
    episode.hist_data['_robot_coordinates'].append(episode.user_data['robot_cur_pos'])


def on_train_result(result):
    """
     Histogram stats are accumulated over iterations, resulting in data from any previous iteration shaping the
    histogram of this iteration. To display the histogram of data only for this iteration any previous is deleted.
    This is performed for custom histograms and RLlib built in histograms as well!!!
    :param result:
    :return:
    """

    episodes_this_iter = result['result']['episodes_this_iter']
    timesteps_this_iter = result['result']['timesteps_this_iter']

    # Custom histograms
    result['result']['hist_stats']['sampled_actions'] = \
        result['result']['hist_stats']['sampled_actions'][:timesteps_this_iter]
    result['result']['hist_stats']['_robot_coordinates'] = \
        result['result']['hist_stats']['_robot_coordinates'][:episodes_this_iter]

    # Built in histograms
    result['result']['hist_stats']['episode_lengths'] = \
        result['result']['hist_stats']['episode_lengths'][:episodes_this_iter]
    result['result']['hist_stats']['episode_reward'] = \
        result['result']['hist_stats']['episode_reward'][:episodes_this_iter]

    # curriculum_apply_update(result)


def enable_obstacles(env):
    semi_unwrapped = env
    while not isinstance(semi_unwrapped, ObstacleSpawningWrapper):
        semi_unwrapped = semi_unwrapped.env
    semi_unwrapped.env_config['spawn_obstacles']=True


def curriculum_apply_update(result):
    """Magic"""
    timesteps_total = result['result'].get('timesteps_total')
    if timesteps_total > 500.e+3 and timesteps_total < 550.e+3:
        trainer = result["trainer"]
        # Alternative: call trainer._make_workers ?
        trainer.workers.foreach_worker_with_index(
            lambda worker, index: worker.foreach_env(lambda env: enable_obstacles(env)))
        logger.warning("Obstacle spawning enabled at timestep {}".format(timesteps_total))

