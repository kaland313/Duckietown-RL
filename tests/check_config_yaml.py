from ray.tune.logger import pretty_print
from config.config import load_config, print_config

cfg = load_config("./config/config.yml")
print_config(cfg)
print("seed:", type(cfg["seed"]), cfg["seed"])
print("rllib_config:monitor:", type(cfg["rllib_config"]["monitor"]))
print("rllib_config:lr:", type(cfg["rllib_config"]["lr"]))
print("timesteps_total:",  type(cfg["timesteps_total"]))
print("env_config:resized_input_shape:", type(cfg["env_config"]["resized_input_shape"]))