import torch
import numpy as np
from gym.utils import EzPickle
from mbrl import torch_helpers


from mbrl.environments.abstract_environments import MaskedGoalSpaceEnvironmentInterface
from mbrl.environments.isaacsim import IsaacSimGroundTruthSupportEnv
from mbrl.environments.isaacgymenv.isaacgymenv import IsaacGymEnv


class IsaacGym(MaskedGoalSpaceEnvironmentInterface, IsaacSimGroundTruthSupportEnv, IsaacGymEnv):
    def __init__(self, *, name, **kwargs):
        IsaacSimGroundTruthSupportEnv.__init__(self, name=name, **kwargs)   
        IsaacGymEnv.__init__(self, name=name, **kwargs)
        EzPickle.__init__(self, name=name, **kwargs)
        
        # 
        MaskedGoalSpaceEnvironmentInterface.__init__(self, name=name, goal_idx=self.ig_env.goal_idx, achieved_goal_idx=self.ig_env.goal_idx, sparse=False, threshold=0.1)
        # MaskedGoalSpaceEnvironmentInterface.__init__(self, name, self.ig_env.goal_idx, self.ig_env.achieved_goal_idx, sparse=False, threshold=0.1)

        self.observation_space_size_preproc = self.obs_preproc(np.zeros(self.ig_env.observation_space.shape[0])).shape[0]

    def viewer_setup(self):
        self.viewer_setup(self)

    # TODO: Implement the method for setting the ground truth state
    def set_GT_state(self, state):
        pass

    # TODO: Implement the method for getting the ground truth state
    def get_GT_state(self):
        return None

    def set_state_from_observation(self, observation):
        raise NotImplementedError

    def gripper_pos_to_target_distance(self, gripper_pos, target_pos):
        raise NotImplementedError

    def cost_fn(self, observation, action, next_obs):
        rew = self.ig_env.compute_reward_sas(observation, action, next_obs)
        return -rew

    def targ_proc(self, observations, next_observations):
        return next_observations - observations
    
    def obs_preproc(self, observation):
        self.obs_shape = observation.shape
        self.goal = observation[..., self.goal_idx]
        return self.observation_wo_goal(observation=observation)
    
    def obs_postproc(self, obs, pred=None, out=None):
        if pred is not None:
            obs = obs + pred
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32).to(torch_helpers.device)
        if not torch.is_tensor(self.goal):
            self.goal = torch.tensor(self.goal, dtype=torch.float32).to(torch_helpers.device)
        goal_tensor = self.goal.clone().to(torch_helpers.device)
        res = torch.zeros(self.obs_shape).to(torch_helpers.device)
        mask = torch.zeros(self.obs_shape[-1], dtype=torch.bool)
        mask[self.goal_idx] = True
        res[..., mask] = goal_tensor
        res[..., ~mask] = obs
        return res
    

    def get_object_centric_obs(self, obs, agent_dim=24, object_dim=13, object_static_dim=6):
        # TODO: Check if this is implemented correctly
        """Preprocessing on the observation to make the input suitable for GNNs

        :param obs: N x (nA + nO * nFo + n0 * nSo) Numpy array
        :param agent_dim: State dimension for the agent
        :param object_dim: State dimension for a single object
        """
        return self.ig_env.compute_object_centric_observation(obs, agent_dim, object_dim, object_static_dim)