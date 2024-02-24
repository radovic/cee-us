import numpy as np
from gym.utils import EzPickle

from mbrl.environments.isaacsim import IsaacSimGroundTruthSupportEnv
from mbrl.environments.isaacgymenv.isaacgymenv import IsaacGymEnv

class IsaacGym(IsaacSimGroundTruthSupportEnv, IsaacGymEnv):
    def __init__(self, *, name, **kwargs):
        IsaacSimGroundTruthSupportEnv.__init__(self, name=name, **kwargs)   
        IsaacGymEnv.__init__(self, name=name, **kwargs)
        EzPickle.__init__(self, name=name, **kwargs)

        # TODO: Implement the following
        self.agent_dim = 3
        self.object_dyn_dim = 3
        self.object_stat_dim = 0
        self.nObj = 2

        self.observation_space_size_preproc = self.obs_preproc(np.zeros(self.ig_env.observation_space.shape[0])).shape[0]

    def viewer_setup(self):
        self.viewer_setup(self)

    def set_GT_state(self, state):
        self.ig_env.gym.set_sim_params(self.ig_env.sim, state)
        self.ig_env.gym.simulate(self.ig_env.sim)

    def get_GT_state(self):
        return self.ig_env.gym.get_sim_params(self.ig_env.sim)

    def set_state_from_observation(self, observation):
        raise NotImplementedError

    def gripper_pos_to_target_distance(self, gripper_pos, target_pos):
        raise NotImplementedError

    def cost_fn(self, observation, action, next_obs):
        # TODO: Check if this is implemented correctly
        rew = self.ig_env.compute_reward(observation, action, next_obs)
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

    @staticmethod
    def get_object_centric_obs(obs, agent_dim=24, object_dim=13, object_static_dim=6):
        """Preprocessing on the observation to make the input suitable for GNNs

        :param obs: N x (nA + nO * nFo + n0 * nSo) Numpy array
        :param agent_dim: State dimension for the agent
        :param object_dim: State dimension for a single object
        """
        raise NotImplementedError("Getting object-centric observations is not implemented yet.")