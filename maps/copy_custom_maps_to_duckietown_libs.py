import os
from pathlib import Path
import gym_duckietown
import duckietown_world


def copy_custom_maps(src, dst):
    assert src.exists(), "Source dir for custom maps maps folder not fund at {}".format(src)
    assert dst.exists(), "Destination dir for custom maps maps folder not fund at {}".format(dst)
    print("\nCopying custom maps to {}".format(dst))
    os.system('cp -arv {}/*.yaml {}/'.format(src, dst))


custom_maps_path = Path(__file__).resolve().parent
print("Path of custom maps: \t\t\t\t {}".format(custom_maps_path))
gym_path = Path(gym_duckietown.__file__).parent
print("gym_duckietown is imported from: \t {}".format(gym_path))
dtworld_path = Path(duckietown_world.__file__).parent
print("duckietown_world is imported from: \t {}".format(dtworld_path))

gym_maps_path = gym_path.joinpath('maps')
copy_custom_maps(custom_maps_path, gym_maps_path)

dtworld_maps_path = dtworld_path.joinpath('data', 'gd1', 'maps')
copy_custom_maps(custom_maps_path, dtworld_maps_path)