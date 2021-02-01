"""
Method that plots multi-episode trajectories, recorded on multiple maps to a single figure as many dotted lines.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, patches as patches
from matplotlib.path import Path


def correct_gym_duckietown_coordinates(sim, pos):
    """
    Gym duckietown uses a weird, coordinate system where the pos[2], z direction is flipped, and shifted.
    Duckietown world usese the unflipped version, this function handles this correction.
    The calculation is based on gym_duckietown.simulator.Simulator.cartesian_from_weird(pos, angle)
    :param sim: gym_duckietown.simulator.Simulator
    :param pos:
    :return:
    """
    # cartesian_position_SE2 = sim.cartesian_from_weird(pos, angle)
    # transform = cartesian_position_SE2.transform_values(SE2Transform.from_SE2)  #type: SE2Transform
    return np.array([pos[0], sim.grid_height * sim.road_tile_size - pos[2]])


def plot_trajectories(paths, rewards=None, show_plot=False, road_tile_size=0.585, unify_start_tile=True):
    if show_plot:
        plt.switch_backend('tkagg')
    fig = plt.figure()
    ax = fig.gca()
    for i, episode_path in enumerate(paths):
        episode_path = np.stack(episode_path)
        if unify_start_tile:
            init_pos = episode_path[0]
            init_tile_coords = np.floor(init_pos / road_tile_size) * road_tile_size
            episode_path -= init_tile_coords
        if rewards is not None:
            plt.scatter(episode_path[:, 0], episode_path[:, 1], s=0.1, c=rewards[i])
        else:
            plt.scatter(episode_path[:, 0], episode_path[:, 1], s=0.1)
        plt.axis('equal')
        plt.plot([episode_path[0, 0]], [episode_path[0, 1]], 'rx')
    if unify_start_tile:
        tile_borders = Path(np.array([[0., 0.],
                                      [0., road_tile_size],
                                      [road_tile_size, road_tile_size],
                                      [road_tile_size, 0.],
                                      [0., 0.]]))
        patch = patches.PathPatch(tile_borders, facecolor='none', lw=2, linestyle='--')
        ax.add_patch(patch)
    ax.axis('off')
    plt.axis('equal')
    if rewards is not None:
        plt.colorbar(shrink=0.5)
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig