from abc import ABC
import numpy as np

from isaacgymenvs.tasks.base.vec_task import VecTask

from mbrl.environments.abstract_environments import GroundTruthSupportEnv


class IsaacSimGroundTruthSupportEnv(ABC):
    """adds generic state operations for all IsaacSim-based envs"""

    def __init__(self, *, name, **kwargs):
        # super().__init__(name=name, **kwargs)
        self.init_kwargs = kwargs
        self.goal_state = None
        self.goal_mask = None
        self.supports_live_rendering = True

    # === Taken from base_types ===
    def prepare_for_recording(self):
        pass

    # === Taken from GroundTruthSupportEnv ===
    def store_init_arguments(self, all_parameters):
        # hacky way to store the parameters that are used to construct the object
        # (which we need to create a copy without a copy operation, namely by called the constructor again)
        forbidden_parameters = ["name", "self", "__class__", "kwargs"]
        self.init_kwargs.update({k: v for k, v in all_parameters.items() if k not in forbidden_parameters})
        if "kwargs" in all_parameters:
            self.init_kwargs.update(
                {k: v for k, v in all_parameters["kwargs"].items() if k not in forbidden_parameters}
            )

    # noinspection PyMethodMayBeStatic
    def compute_state_difference(self, state1, state2):
        return np.max(state1 - state2)

    # simulates one step of the env by resetting to the given state first (not
    # the observation, but the env-state)
    def simulate(self, state, action):
        self.set_GT_state(state)
        new_obs, r, *_ = self.step(action)
        new_state = self.get_GT_state()
        return new_obs, new_state, r
    
    # === Implemented specifically for IsaacGym ===
    # noinspection PyPep8Naming
    def set_GT_state(self, state):
        self.ig_env.gym.set_sim_params(self.ig_env.sim, state)
        self.ig_env.gym.simulate(self.ig_env.sim)

    # noinspection PyPep8Naming
    def get_GT_state(self):
        return self.ig_env.gym.get_sim_params(self.ig_env.sim)

    # noinspection PyMethodMayBeStatic
    def prepare_for_recording(self):
        # TODO: Implement this
        pass

    # === Taken from EnvWithDefaults ===
    # noinspection PyUnusedLocal
    def cost_fn(self, observation, action, next_obs):
        # compute for all samples, along coordinate axis
        dist = np.linalg.norm((observation - self.goal_state) * self.goal_mask, axis=-1)
        return dist

    def reward_fn(self, observation, action, next_obs):
        return -self.cost_fn(observation, action, next_obs)

    def from_full_state_to_transformed_state(self, full_state):
        return full_state

    def reset_with_mode(self, mode):
        return self.reset()

    def get_fps(self):
        if hasattr(self.ig_env, "dt"):
            return int(np.round(1.0 / self.ig_env.dt))
        elif hasattr(self, "metadata") and "video.frames_per_second" in self.metadata:
            return self.metadata["video.frames_per_second"]
        else:
            raise NotImplementedError("Environment does not have a generic way to get FPS. Overwrite get_fps()")

    @staticmethod
    def targ_proc(observations, next_observations):
        return next_observations - observations

    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def obs_postproc(obs, pred, out=None):
        return obs + pred

    @staticmethod
    def filter_buffers_by_cost(buffers, costs, filtered_fraction):
        if filtered_fraction == 1:
            print("Trajectories are not pre-filtered.")
            return [buffer.flat for buffer in buffers]
        else:
            num = int(len(costs) * filtered_fraction)
            print(f"Pre-filtering (keeping) {filtered_fraction * 100:.2f}% of all trajectories in the memory.")
            idxs = [np.array(c["costs"]).argsort()[:num] for c in costs]
            return [buffer.flat[idx] for buffer, idx in zip(buffers, idxs)]