import yaml
from argparse import Namespace
from pathlib import Path

def add_args_from_config(args: Namespace) -> Namespace:
    """
    Read .yml config file and add fields to args
    """
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        if isinstance(config[key],dict):
            for k, v in config[key].items():
                setattr(args, k, format_value(key,k,v))
        else:
            setattr(args, key, config[key])

    return args


def format_value(key,k,v):
    if key == "paths" or any(k.endswith(x) for x in ["_path","_dir"]):
        return Path(v)
    return v