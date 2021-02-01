"""
Draw top-down visualizations of a map-set or a single map
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"


from duckietown_world.svg_drawing.draw_maps import draw_map
from duckietown_world.world_duckietown import list_maps, load_map

from duckietown_utils.env import MAPSETS

# Draw every map in a mapset (see  duckietown_utils.env -> MAPSETS for the list of mapsets)
for map in MAPSETS['multimap_aido5']:
    duckietown_map = load_map(map)
    draw_map('maps/' + map, duckietown_map)

# Draw a single map
map_name = '_myTestA'
duckietown_map = load_map(map_name)
draw_map('maps/' + map_name, duckietown_map)