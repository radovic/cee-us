import copy
import isaacgym
import torch
import random
from collections import namedtuple

from mbrl.environments.isaacgymenv.loader import load_isaacgym_env

import numpy as np
from gym import utils

from mbrl import seeding

# Code taken and modified to mujoco_py from: https://github.com/google-research/robodesk
class IsaacGymEnv(utils.EzPickle):
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
        } # TODO: Check if needed

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

    '''
    def set_state(self):
        raise NotImplementedError
    '''

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

    def step(self, action):
        action = torch.Tensor([action] * self.num_envs).to(self.ig_env.device) # TODO: TEMPORARY: Clone the action for all environments
        obs, rew, done, info = self.ig_env.step(action)
        return (obs['obs'][0].cpu().detach().numpy(), rew[0].cpu().detach().numpy(), done[0].cpu().detach().numpy(), {}) # TODO: TEMPORARY: Only return the first observation

    def compute_reward(self):
        return self.ig_env.compute_reward()

    def reset(self):
        return self.ig_env.reset()['obs'][0].cpu().detach().numpy() # TODO: TEMPORARY: Only return the first observation

    def _get_obs(self):
        return self.ig_env.compute_observations()['obs'][0].cpu().detach().numpy() # TODO: TEMPORARY: Only return the first observation
    
    def render(self, mode="human"):
        self.ig_env.render(mode=mode)

    def close(self):
        self.ig_env.close()


if __name__ == "__main__":
    env = IsaacGymEnv(task="FrankaCubeStack", num_envs=2, headless=False, isaacgymenvs_path="")
    env.reset()
    for t in range(1000):
        if t % 10 == 0:
            action = [env.action_space.sample(), env.action_space.sample()]
        obs, r, d, i = env.step(action)
        env.render()
    env.close()
