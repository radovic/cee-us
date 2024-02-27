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

class TorchMpcMPPI(MpcController):
    """
        Model Predivtive Path Integral https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf

        Controller computes cost for candidate trajectories around a (mean) nominal action tensor self.u_.
        The update to the mean tensor is a weighted sum of the trajectories. At the end of each
        get_action call, we shift self.u_ to the left, effectively using our pre-computed actions as the new mean vector
        to generate trajectories around.
    
    """
    def __init__(self, *, action_sampler_params, use_async_action, logging=True, fully_deterministic=False, **kwargs):

        super().__init__(**kwargs)
        self._parse_action_sampler_params(**action_sampler_params)
        self._check_validity_parameters()

        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        self.last_action = None
        self.was_reset = False
        self.logging = logging
        # In fully deterministic setup controller will use stable sorting, but it might lead to slow down
        self.fully_deterministic = fully_deterministic

        self.use_async_action = use_async_action
        self._ensemble_size = getattr(self.forward_model, "ensemble_size", None)
        self.a_dim = self.env.action_space.shape[0]

        if not isinstance(self.forward_model, TorchModel):
            warnings.warn(
                "Forward model is not TorchModel, wrapping for compatibility, this might cause things to become slow!"
            )
            if isinstance(self.forward_model, (GroundTruthModel, ParallelGroundTruthModel)):
                self.forward_model = Torch2NumpyGroundTruthModelWrapper(self.forward_model)
                self.forward_model.ensemble_size = getattr(self.forward_model, "ensemble_size", None)
            else:
                raise Exception(f"Model {self.forward_model} not supported!")

        if self._ensemble_size:
            self.forward_model.num_simulated_trajectories = self.num_sim_traj
            self.forward_model.horizon = self.horizon
            if hasattr(self.forward_model, "preallocate_memory"):
                self.forward_model.preallocate_memory()

        if hasattr(self.forward_model, "get_state"):
            state = self.forward_model.get_state()
            if state is None:
                self.state_dim = None
            else:
                self.state_dim = state.shape[-1]
        else:
            self.state_dim = None

        self.preallocate_memory()
        self.cost_along_trajectory = 'sum'

    @torch.no_grad()
    def preallocate_memory(self):
        """
        Preallocated tensors end with and underscore
        Use in-place operations, i.e. use tensor operations with out to
        specify the destination for efficiency
        """

        # nominal actions buffer
        self.u_ = torch.zeros(self.dim_samples, device=torch_helpers.device, dtype=torch.float32)
        
        # standart deviation of error
        self.std_ = torch.ones(self.dim_samples, device=torch_helpers.device, dtype=torch.float32)

        self.action_high_tensor = torch.zeros_like(self.u_, device=torch_helpers.device, dtype=torch.float32)
        self.action_high_tensor[..., :] = torch.from_numpy(self.env.action_space.high).float().to(torch_helpers.device)
        self.action_low_tensor = torch.zeros_like(self.u_, device=torch_helpers.device, dtype=torch.float32)
        self.action_low_tensor[..., :] = torch.from_numpy(self.env.action_space.low).float().to(torch_helpers.device)

        self.delta_u_ = torch.zeros(
            self.num_sim_traj,
            *self.u_.shape,
            device=torch_helpers.device,
            dtype=torch.float32,
        )

        if self.state_dim is not None:
            self.start_states_ = torch.empty((self.num_sim_traj, self.state_dim))
        else:
            self.start_states_ = [None] * self.num_sim_traj

        if self._ensemble_size:
            self.costs_per_model_ = torch.zeros(
                (self.num_sim_traj, self._ensemble_size),
                device=torch_helpers.device,
                dtype=torch.float32,
            )
            self.costs_ = torch.zeros(self.num_sim_traj, device=torch_helpers.device, dtype=torch.float32)
            self.costs_std_ = torch.zeros(self.num_sim_traj, device=torch_helpers.device, dtype=torch.float32)
        else:
            self.costs_ = torch.zeros(self.num_sim_traj, device=torch_helpers.device, dtype=torch.float32)

    @torch.no_grad()
    def set_init_action(self, action):
        self.last_action = torch.from_numpy(action).float().to(torch_helpers.device)  # .astype(np.float32)

    @torch.no_grad()
    def trajectory_cost_fn(self, cost_fn, rollout_buffer: RolloutBuffer, out: torch.Tensor):
        if self.use_env_reward:
            raise NotImplementedError()
        else:
            costs_path = cost_fn(
                rollout_buffer.as_array("observations"),
                rollout_buffer.as_array("actions"),
                rollout_buffer.as_array("next_observations"),
            )  # shape: [p,h]

        # Watch out: result is written to preallocated variable 'out'
        if self.cost_along_trajectory == "sum":
            return torch.sum(costs_path, axis=-1, out=out)
        elif self.cost_along_trajectory == "best":
            return torch.amin(costs_path[..., 1:], axis=-1, out=out)
        elif self.cost_along_trajectory == "final":
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                "Implement method {} to compute cost along trajectory".format(self.cost_along_trajectory)
            )

    @torch.no_grad()
    def beginning_of_rollout(self, *, observation, state=None, mode):
        super().beginning_of_rollout(observation=observation, state=state, mode=mode)

    @torch.no_grad()
    def end_of_rollout(self, total_time, total_return, mode):
        super().end_of_rollout(total_time, total_return, mode)

    def sample_action_sequences(self):
    
        # (num_trajs, horizon_n, a_dim)
        torch.randn(size=self.delta_u_.shape,
                              device=torch_helpers.device, dtype=torch.float32, out=self.delta_u_)
        
        # multiply with std, write back into self.delta_u_
        torch.mul(self.delta_u_, self.std_, out=self.delta_u_)

        # broadcast the deltas onto the nominal actions.
        action_sequences = self.delta_u_ + self.u_[None, ...]
        
        # clip for legal action range
        torch.min(action_sequences, self.action_high_tensor, out=action_sequences)
        torch.max(action_sequences, self.action_low_tensor, out=action_sequences)

        return action_sequences
    
    @torch.no_grad()
    def get_action(self, obs, state, mode="train"):
        """
            Plans for the next action `self.horizon` steps into the future. 

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

        # sample action sequences v around the nominal (mean) action sequence u
        action_sequences = self.sample_action_sequences()
        
        # repeat the obs to match num_trajs. shape: (num_trajs, o_dim)
        obs_ = torch.atleast_2d(torch.from_numpy(obs)).repeat(self.num_sim_traj, 1).to(torch_helpers.device)

        # Monte Carlo Simulation 
        rollouts = self.simulate_trajectories(obs=obs_, 
                                              state=self.forward_model_state, 
                                              action_sequences=action_sequences)

        # writes the costs into pre-allocated buffer self.costs_.
        # In case we have ensembles, we 
        if self._ensemble_size:
                self.trajectory_cost_fn(
                    self.cost_fn, rollouts, out=self.costs_per_model_
                )  # shape [num_sim_traj, num_models]

                torch.mean(self.costs_per_model_, -1, out=self.costs_)
                # could be used to weigh the costs
                torch.std(self.costs_per_model_, -1, out=self.costs_std_)

                if self.use_ensemble_cost_std:
                    torch.add(self.costs_, self.costs_std_, out=self.costs_)
        else:
            self.trajectory_cost_fn(self.cost_fn, rollouts, out=self.costs_)  # shape: [num_sim_paths]
            
        # calculate weighting. The less cost a action sequence has accumulated, the 
        # heavier its influence on the best action sequence should be.
        torch.subtract(self.costs_, self.costs_.min(0)[0], out=self.costs_)
        torch.mul(self.costs_, 1/self.temperature)

        w = torch.softmax(-self.costs_, dim=0)

        # mult w onto self.delta_u_, then add to the nominal u. 
        # This needs to follow broadcasting rules, so we need
        # to expand w's dimensions accordingly
        torch.add(self.u_, torch.sum(self.delta_u_ * w[:, None, None], dim=0), out=self.u_)

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

    @torch.no_grad()
    def simulate_trajectories(self, *, obs, state, action_sequences: torch.tensor) -> RolloutBuffer:
        """
        :param obs: current starting observation
        :param state: current starting state of forward model
        :param action_sequences: shape: [p,h,d]
        """
        with torch.no_grad():
            if state is not None:
                self.start_states_[:] = to_tensor(state[None]).to(torch_helpers.device)
            else:
                self.start_states_ = [None] * self.num_sim_traj

            return self.forward_model.predict_n_steps(
                start_observations=obs,
                start_states=self.start_states_,
                policy=OpenLoopPolicy(action_sequences),
                horizon=self.horizon,
            )[0]
        
    def _parse_action_sampler_params(
        self,
        *,
        alpha,
        elites_size,
        opt_iterations,
        init_std,
        use_mean_actions,
        keep_previous_elites,
        shift_elites_over_time,
        finetune_first_action,
        fraction_elites_reused,
        colored_noise,
        noise_beta=1,
        relative_init,
        execute_best_elite,
        use_ensemble_cost_std,
        temperature,
        min_std,
        max_std
    ):

        self.alpha = alpha
        self.elites_size = elites_size
        self.opt_iter = opt_iterations
        self.init_std = init_std
        self.use_mean_actions = use_mean_actions
        self.keep_previous_elites = keep_previous_elites
        self.shift_elites_over_time = shift_elites_over_time
        self.fraction_elites_reused = fraction_elites_reused
        self.finetune_first_action = finetune_first_action
        self.colored_noise = colored_noise
        self.noise_beta = noise_beta
        self.relative_init = relative_init
        self.execute_best_elite = execute_best_elite
        self.use_ensemble_cost_std = use_ensemble_cost_std
        self.temperature = temperature
        self.min_std = min_std
        self.max_std = max_std


    def _check_validity_parameters(self):

        self.num_elites = min(self.elites_size, self.num_sim_traj // 2)
        if self.num_elites < 2:
            warn("Number of trajectories is too low for given elites_frac. Setting num_elites to 2.")
            self.num_elites = 2

        if isinstance(self.env.action_space, spaces.Discrete):
            raise NotImplementedError("CEM ERROR: Implement categorical distribution for discrete envs.")
        elif isinstance(self.env.action_space, spaces.Box):
            self.dim_samples = (self.horizon, self.env.action_space.shape[0])
        else:
            raise NotImplementedError