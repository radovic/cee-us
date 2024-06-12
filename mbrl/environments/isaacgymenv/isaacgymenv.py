import copy
import isaacgym
import torch
import random
from collections import namedtuple

from mbrl.environments.isaacgymenv.loader import load_isaacgym_env

import numpy as np
from gym import utils

from mbrl import seeding
from typing import Tuple

# Code taken and modified to mujoco_py from: https://github.com/google-research/robodesk
class IsaacGymEnv(utils.EzPickle):
    """
    This class is a wrapper for an Isaac Gym environment. 
    It initializes an Isaac Gym environment with specific settings (task, number 
    of environments, headless mode, path to environment files).
    """
    def __init__(
        self,
        name="IsaacGymEnv",
        task="FrankaCubeStack",
        num_envs=5,
        headless=False,
        isaacgymenvs_path="",
        show_cfg=False,
    ):
        ''' 
        Serves as a wrapper between the isaacgym environment and the rest of the code 
        '''
        self.num_envs = num_envs
        self.ig_env = load_isaacgym_env(task=task, num_envs=num_envs, headless=headless, isaacgymenvs_path=isaacgymenvs_path, show_cfg=show_cfg)

        self.obs_type = "state" # TODO: Check if needed
        self.viewer = None # TODO: Check if needed
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
        }

        # TODO: improve this
        # Get body names
        self.agent_dim, self.object_dyn_dim, self.object_stat_dim, self.nObj = self.ig_env.get_object_dims()
        self.env_body_names = [f'cube{i}' for i in range(self.nObj)]

        self.initial_state = copy.deepcopy(self.ig_env.gym.get_sim_params(self.ig_env.sim))

        # Environment params
        self.action_dim = self.ig_env.action_space.shape[0]

        obs = self.ig_env.compute_observations()

        self.action_space = self.ig_env.action_space
        self.observation_space = self.ig_env.observation_space

        self.original_pos = {} # TODO: Check if needed
        self.previous_z_angle = None # TODO: Check if needed
        self.total_rotation = 0 # TODO: Check if needed

        self.task = task

        utils.EzPickle.__init__(
            self,
            task,
            num_envs,
            headless,
            isaacgymenvs_path,
        )

    def set_state(self):
        raise NotImplementedError('Setting state is not implemented yet')

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    @property
    def dt(self):
        # TODO: Check if this is the correct timestep
        return self.ig_env.sim.model.opt.timestep * self.ig_env.sim.nsubsteps
    
    def _get_task_reward(self):
        return self.ig_env.compute_reward

    def step(self, action : np.array) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Takes an action in the environment and returns the next observation, reward, 
            done flag, and info dictionary.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the following elements:
                obs (dict): The new observation from the environment.
                rew (float): The reward for taking this action.
                done (bool): Whether the episode has ended.
                info (dict): Any additional information about the step, such as 
                                whether the episode ended.
        """

        
        action = torch.Tensor(action).to(self.ig_env.device)
        obs, rew, done, info = self.ig_env.step(action)

        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.cpu().detach().numpy()
                
        return (obs['obs'].cpu().detach().numpy(), rew.cpu().detach().numpy(), done.cpu().detach().numpy(), info)

    def compute_reward(self):
        return self.ig_env.compute_reward()

    def reset(self):
        self.ig_env.reset_idx(torch.arange(self.num_envs, device=self.ig_env.device))
        return self._get_obs()

    def _get_obs(self):
        return self.ig_env.compute_observations().cpu().detach().numpy()
    
    def render(self, mode="human"):
        self.ig_env.render(mode=mode)

    def close(self):
        pass


if __name__ == "__main__":
    env = IsaacGymEnv(task="FrankaCubeStack", num_envs=2, headless=False, isaacgymenvs_path="")
    env.reset()
    for t in range(1000):
        if t % 10 == 0:
            action = [env.action_space.sample(), env.action_space.sample()]
        obs, r, d, i = env.step(action)
        env.render()
    env.close()
