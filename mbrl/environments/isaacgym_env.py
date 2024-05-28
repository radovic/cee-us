import numpy as np
from gym.utils import EzPickle

from mbrl.environments.isaacsim import IsaacSimGroundTruthSupportEnv
from mbrl.environments.isaacgymenv.isaacgymenv import IsaacGymEnv

class IsaacGym(IsaacSimGroundTruthSupportEnv, IsaacGymEnv):
    def __init__(self, *, name, **kwargs):
        IsaacSimGroundTruthSupportEnv.__init__(self, name=name, **kwargs)   
        IsaacGymEnv.__init__(self, name=name, **kwargs)
        EzPickle.__init__(self, name=name, **kwargs)

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
        return observation

    def obs_postproc(self, obs, pred=None, out=None):
        if pred is not None:
            return obs + pred
        else:
            return obs

    def get_object_centric_obs(self, obs, agent_dim=24, object_dim=13, object_static_dim=6):
        # TODO: Check if this is implemented correctly
        """Preprocessing on the observation to make the input suitable for GNNs

        :param obs: N x (nA + nO * nFo + n0 * nSo) Numpy array
        :param agent_dim: State dimension for the agent
        :param object_dim: State dimension for a single object
        """
        return self.ig_env.compute_object_centric_observation(obs, agent_dim, object_dim, object_static_dim)