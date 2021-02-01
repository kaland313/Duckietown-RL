import yaml
from pathlib import Path
import numpy as np

custom_maps_path = Path(__file__).resolve().parent

maps = ["LF-norm-zigzag.yaml", "LF-norm-techtrack.yaml", "LF-norm-small_loop.yaml", "LF-norm-loop.yaml"]
for mapstr in maps:
    print(mapstr)
    with open(custom_maps_path.joinpath(mapstr), 'w+') as f:
        yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
        tile_size = yaml_data['tile_size']
        map_height = len(yaml_data['tiles']) ## number of rows, measured in tiles
        obj_list = []
        if isinstance(yaml_data["objects"], list):
            print(mapstr, "already processed")
            continue
        for key, val in yaml_data["objects"].items():
            obj_list.append({key: None,
                             **val})
        for obj in obj_list:
            if 'place' in obj.keys():
                obj['rotate'] = obj['place']['relative']['~SE2Transform']['theta_deg']
                x, y = obj['place']['relative']['~SE2Transform']['p']
                i, j = obj['place']['tile']
                u = (x + i + 0.5)
                v = map_height - (y + j + 0.5)
                obj['pos'] = [u, v]
                del obj['place']
            elif 'pose' in obj.keys():
                if 'theta' in obj['pose']['~SE2Transform'].keys():
                    obj['rotate'] = float(np.rad2deg(obj['pose']['~SE2Transform']['theta']))
                else:
                    obj['rotate'] = float(obj['pose']['~SE2Transform']['theta_deg'])
                obj['pos'] = (np.array(obj['pose']['~SE2Transform']['p'])/tile_size).tolist()
                x, y = (np.array(obj['pose']['~SE2Transform']['p'])/tile_size).tolist()
                u = x
                v = map_height - y
                obj['pos'] = [u, v]
                del obj['pose']

        yaml_data['objects'] = obj_list
        f.seek(0)
        yaml.dump(yaml_data, f, Dumper=yaml.SafeDumper)
        f.truncate()