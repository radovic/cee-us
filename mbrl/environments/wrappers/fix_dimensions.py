from .abstract import EnvWrapper
import numpy as np

class SingleEnvironmentWrapper(EnvWrapper):
    """
        Wrapper to reduce the dimension of inputs to
        meet requirements of single env simulations, while
        expanding the dimensions to include an explicit num_envs dimension=1
        to let the vectorized controllers work as expected.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action : np.array):
        action = action.squeeze(0) # flatten the env dimension
        next_ob, rew, done, info_dict = self.env.step(action)
        
        # create the env-dimension at position 0
        next_ob = np.expand_dims(next_ob, 0)
        rew = np.expand_dims(rew, 0)
        done = np.expand_dims(done, 0)

        return next_ob, rew, done, info_dict

    def set_state_from_observation(self, ob):
        self.env.set_state_from_observation(ob.squeeze(0))
    
    def set_GT_state(self, state):
        self.env.set_GT_state(state.squeeze(0))
    
    def reset_with_mode(self, mode):
        ob = self.env.reset_with_mode(mode)
        return np.expand_dims(ob, 0)

    @property
    def inital_action(self):
        return np.expand_dims(self.env.initial_action, 0)


    