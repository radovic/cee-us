from mbrl.controllers.mpc_mppi_torch import TorchMpcMPPI
import torch

from mbrl import torch_helpers

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
        if not self.was_reset:
            raise AttributeError("beginning_of_rollout() needs to be called before")
        
        self.forward_model_state = self.forward_model.got_actual_observation_and_env_state(
            observation=obs, env_state=state, model_state=self.forward_model_state
        )

        # repeat the obs to match num_trajs. shape: (num_envs, num_trajs, o_dim)
        # obs has shape (num_env, o_dim) -> unsqueeze to (num_envs, 1 , o_dim), repeat
        obs_ = torch.from_numpy(obs)\
            .unsqueeze(1)\
            .repeat(1, self.num_sim_traj, 1)\
            .to(torch_helpers.device)
        

        # iterate and find good action sequences. Keep track of elite action seqs.
        for i_iter in range(self.opt_iter):

            # sample action sequences
            action_sequences = self.sample_action_sequences()
        
            # Monte Carlo Simulation -> roll out the forward model given the action sequences.
            rollouts = self.simulate_trajectories(obs=obs_, 
                                                state=self.forward_model_state, 
                                                action_sequences=action_sequences)

            # writes the costs into pre-allocated buffer self.costs_.
            # self.costs_ will have shape [num_envs, num_sim_traj]
            if self._ensemble_size:
                    self.trajectory_cost_fn(
                    self.cost_fn, rollouts, out=self.costs_per_model_
                    )  # shape [num_envs, num_sim_traj, num_models]

                    # TODO: mean over which dimension? -> Should sum out the ensemble dimension.
                    torch.mean(self.costs_per_model_, -1, out=self.costs_) # shape: [num_envs, self.num_sim_traj]
                    # could be used to weigh the costs
                    torch.std(self.costs_per_model_, -1, out=self.costs_std_)

                    if self.use_ensemble_cost_std:
                        torch.add(self.costs_, self.costs_std_, out=self.costs_)
            else:
                self.trajectory_cost_fn(self.cost_fn, rollouts, out=self.costs_)

            
            
            # self.costs_ shape [num_envs, num_sim_traj]
            # find the top self.num_elites action sequences with lowest cost.
            elite_rewards, elite_indices = torch.topk(-self.costs_, self.num_elites, dim=1) # shape: [num_envs, num_elites]
            
            # expand dimensions of indices [num_envs, num_elites] -> [num_envs, num_elites, h, a_dim]
            elite_indices = elite_indices[..., None, None].expand(*elite_indices.shape, *action_sequences.shape[-2:])
            elite_actions = torch.gather(input=action_sequences,
                                         dim=1,
                                         index=elite_indices)
            
            max_reward = elite_rewards.max(-1).values # compute the max along the elites dimension.

            # calculate weighting for the elite action sequences.
            # The more reward an action sequence has accumulated, the 
            # heavier its influence on the nominal (==mean) action sequence should be.
            # The weighting is a softmax over the rewards
            torch.subtract(elite_rewards, max_reward[:, None], out=elite_rewards)
            score  = torch.exp(1/self.temperature * elite_rewards)
            score /= score.sum(-1)[:, None]
            
            # TODO: TDMPC2 implements the mean update as a hard update, i.e. the mean vector is overwritten
            # with the mean of the weighted elite_actions. Usually in MPPI we would apply the 
            # weighted elite stds (scores[:, None, None] * self.delta_u_[elite_idxs]) to the mean vector. Ask Cansu
            # whether or not this is the right implementation. I've also seen variants which use a moving average for the mean...
            self.u_ = torch.sum(elite_actions * score[..., None, None], dim=1)
            self.std_ = torch.sqrt(torch.sum(score[..., None, None] * (elite_actions - self.u_.unsqueeze(1)) ** 2, dim=1))\
                                   .clamp_(self.min_std, self.max_std)
            
        # return best action
        executed_action = self.u_[:, 0]
        executed_action = executed_action.cpu().detach().numpy()
        
        # shift u_ (means) leftwards
        self.u_ = torch.concat([self.u_[:, 1:], 
                         torch.zeros(size=(self.num_envs, 1, self.a_dim), device=torch_helpers.device, dtype=torch.float32)], dim=1)

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

            
    