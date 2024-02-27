from mbrl.controllers.mpc_mppi_torch import TorchMpcMPPI
import warnings
from warnings import warn

import numpy as np
import torch
from gym import spaces

from mbrl import allogger, torch_helpers
from mbrl.controllers import colored_noise
from mbrl.controllers.abstract_controller import OpenLoopPolicy
from mbrl.controllers.mpc import MpcController
from mbrl.models.abstract_models import TorchModel
from mbrl.models.gt_model import GroundTruthModel, Torch2NumpyGroundTruthModelWrapper
from mbrl.models.gt_par_model import ParallelGroundTruthModel
from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer
from mbrl.torch_helpers import to_tensor

class TorchMpcMPPIElites(TorchMpcMPPI):
    """
        MPPI with Elites and multiple iterations. The update to the nominal action sequence self.u_
        is currently implemented as a hard update, i.e. we overwrite self.u_ as the weighted sum of 
        the elite action sequences, instead of updating with the weighted errors. 
    """
    def __init__(self, *, action_sampler_params, use_async_action, logging=True, fully_deterministic=False, **kwargs):
        super().__init__(action_sampler_params=action_sampler_params, use_async_action=use_async_action, logging=logging, fully_deterministic=fully_deterministic, **kwargs)

    def get_action(self, obs : torch.Tensor, state : torch.Tensor, mode : str = "train") -> torch.Tensor:
        """
            Plans for self.opt_iter iterations, keeping track of elite action sequences. Returns the next action.

            Args:

                obs : current observation from the environment. Shape (o_dim,)
                state: current state of the forward model (only used in stateful models). Shape (fwmodel_state_dim,)
                mode: flag to switch between evaluation and training mode.

            Returns:

                executed_action: Action at index 0 of the best found action sequence. Shape (a_dim,)
        """
        self.forward_model_state = self.forward_model.got_actual_observation_and_env_state(
            observation=obs, env_state=state, model_state=self.forward_model_state
        )

        # repeat the obs to match num_trajs. shape: (num_trajs, o_dim)
        obs_ = torch.atleast_2d(torch.from_numpy(obs)).repeat(self.num_sim_traj, 1).to(torch_helpers.device)

        # iterate and find good action sequences. Keep track of elite action seqs.
        for i_iter in range(self.opt_iter):

            # sample action sequences
            action_sequences = self.sample_action_sequences()
        
            # Monte Carlo Simulation -> roll out the forward model given the action sequences.
            rollouts = self.simulate_trajectories(obs=obs_, 
                                                state=self.forward_model_state, 
                                                action_sequences=action_sequences)

            # writes the costs into pre-allocated buffer self.costs_.
            if self._ensemble_size:
                    self.trajectory_cost_fn(
                        self.cost_fn, rollouts, out=self.costs_per_model_
                    )  # shape (num_sim_traj, num_models)

                    torch.mean(self.costs_per_model_, -1, out=self.costs_)
                    torch.std(self.costs_per_model_, -1, out=self.costs_std_)

                    if self.use_ensemble_cost_std:
                        torch.add(self.costs_, self.costs_std_, out=self.costs_)
            else:
                self.trajectory_cost_fn(self.cost_fn, rollouts, out=self.costs_)  # shape: (num_sim_paths)
                    
            # find the top self.num_elites action sequences with lowest cost.
            elite_idxs = torch.topk(-self.costs_, self.num_elites, dim=0).indices
            elite_costs  = self.costs_[elite_idxs]
            elite_actions = action_sequences[elite_idxs]
            min_cost = elite_costs.min(0)[0]

            # calculate weighting for the elite action sequences.
            # The less cost a action sequence has accumulated, the 
            # heavier its influence on the nominal (==mean) action sequence should be.
            # This is just a softmax over the negative elite costs...
            torch.subtract(elite_costs, min_cost, out=elite_costs)
            score  = torch.exp(1/self.temperature * -elite_costs)
            score /= score.sum(0)
            
            # TODO: TDMPC2 implements the mean update as a hard update, i.e. the mean vector is overwritten
            # with the mean of the weighted elite_actions. Usually in MPPI we would apply the 
            # weighted elite stds (scores[:, None, None] * self.delta_u_[elite_idxs]) to the mean vector. Ask Cansu
            # whether or not this is the right implementation. I've also seen variants which use a moving average for the mean...
            self.u_ = torch.sum(elite_actions * score[:, None, None], dim=0)
            self.std_ = torch.sqrt(torch.sum(score[:, None, None] * (elite_actions - self.u_.unsqueeze(0)) ** 2, dim=0))\
                                   .clamp_(self.min_std, self.max_std)
            
        # return best action
        executed_action = self.u_[0]
        executed_action = executed_action.cpu().detach().numpy()
        
        # shift u_ (means) leftwards
        self.u_ = torch.concat([self.u_[1:], 
                         torch.zeros(size=(1, self.a_dim), device=torch_helpers.device, dtype=torch.float32)], dim=0)

        if self.mpc_hook:
            self.mpc_hook.executed_action(obs, executed_action)

        if self.logging:
            self.logger.log(torch.min(self.costs_).item(), key="best_trajectory_cost")

        if self.do_visualize_plan:
            best_traj_idx = torch.argmin(self.costs_)
            viz_obs = rollouts["observations"][best_traj_idx]
            acts = rollouts["actions"][best_traj_idx]
            self.visualize_plan(obs=viz_obs, state=self.forward_model_state, acts=acts)

        # for stateful models, actually simulate step (forward model stores the
        # state internally)
        if self.forward_model_state is not None:
            obs_, self.forward_model_state, rewards = self.forward_model.predict(
                observations=obs,
                states=self.forward_model_state,
                actions=executed_action,
            )

        return executed_action

            
    