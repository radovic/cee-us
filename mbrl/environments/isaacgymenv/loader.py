from typing import Optional, Sequence

import os
import sys
from contextlib import contextmanager

from abc import ABC
from isaacgymenvs.tasks.base.vec_task import VecTask

def _omegaconf_to_dict(config) -> dict:
    """Convert OmegaConf config to dict

    :param config: The OmegaConf config
    :type config: OmegaConf.Config

    :return: The config as dict
    :rtype: dict
    """
    # return config.to_container(dict)
    from omegaconf import DictConfig
    d = {}
    for k, v in config.items():
        d[k] = _omegaconf_to_dict(v) if isinstance(v, DictConfig) else v
    return d

def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration

    :param d: The dictionary to print
    :type d: dict
    :param indent: The indentation level (default: ``0``)
    :type indent: int, optional
    """
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + f"  |-- {key}: {value}")

def load_isaacgym_env(task: str = "",
                        num_envs: Optional[int] = None,
                        headless: Optional[bool] = None,
                        isaacgymenvs_path: str = "",
                        show_cfg: bool = True):

    import isaacgym
    import isaacgymenvs
    import hydra
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_automatic_config_search_path, get_args_parser
    from hydra.types import RunMode
    from omegaconf import OmegaConf

    overrides = []

    # check if task is empty
    if task == "":
        raise ValueError("task argument is required.")
    overrides.append(f"task={task}")
    
    # check if num_envs is empty
    if num_envs is None:
        raise ValueError("num_envs argument is required.")
    overrides.append(f"num_envs={num_envs}")

    # check if headless is empty
    if headless is None:
        raise ValueError("headless argument is required.")
    overrides.append(f"headless={headless}")

    # TODO: Enable custom path for isaacgymenvs
    # get isaacgymenvs path from isaacgymenvs package metadata
    if isaacgymenvs_path == "":
        if not hasattr(isaacgymenvs, "__path__"):
            raise RuntimeError("isaacgymenvs package is not installed")
        isaacgymenvs_path = list(isaacgymenvs.__path__)[0]
    config_path = os.path.join(isaacgymenvs_path, "cfg")

    # set omegaconf resolvers
    try:
        OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver('if', lambda condition, a, b: a if condition else b)
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
    except Exception as e:
        pass

    # get hydra config without use @hydra.main
    hydra.core.global_hydra.GlobalHydra.instance().clear() # re-initialize hydra
    config_file = "config"
    search_path = create_automatic_config_search_path(config_file, None, config_path)
    hydra_object = Hydra.create_main_hydra2(task_name='load_isaacgymenv', config_search_path=search_path)
    config = hydra_object.compose_config(config_file, overrides, run_mode=RunMode.RUN)

    cfg = _omegaconf_to_dict(config.task)

    # print config
    if show_cfg:
        print(f"\nIsaac Gym environment ({config.task.name})")
        _print_cfg(cfg)

    # load environment
    sys.path.append(isaacgymenvs_path)
    from tasks import isaacgym_task_map  # type: ignore
    try:
        env = isaacgym_task_map[config.task.name](cfg=cfg,
                                                  sim_device=config.sim_device,
                                                  graphics_device_id=config.graphics_device_id,
                                                  headless=config.headless)
    except TypeError as e:
        env = isaacgym_task_map[config.task.name](cfg=cfg,
                                                  rl_device=config.rl_device,
                                                  sim_device=config.sim_device,
                                                  graphics_device_id=config.graphics_device_id,
                                                  headless=config.headless,
                                                  virtual_screen_capture=config.capture_video,  # TODO: check
                                                  force_render=config.force_render)
        
    return env