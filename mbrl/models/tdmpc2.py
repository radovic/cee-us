import numpy as np
import torch
import torch.nn.functional as F

from mbrl import torch_helpers
from mbrl.models.tdmpc2_helpers import math
from mbrl.models.tdmpc2_helpers.scale import RunningScale 
from mbrl.models.tdmpc2_helpers.world_model import TDMPC2WorldModel

from mbrl.models.abstract_models import ForwardModel, EnsembleModel, TrainableModel, TorchModel

from mbrl.rolloutbuffer import RolloutBuffer, SimpleRolloutBuffer
from mbrl.controllers.abstract_controller import Controller
from configparser import ConfigParser
from typing import Sequence, Tuple, Optional
from mbrl.helpers import env_name_to_task

class TDMPC2(ForwardModel, EnsembleModel, TrainableModel, TorchModel):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, env, **cfg):
        super().__init__(env=env)
        self._cfg = cfg
        self.cfg = cfg['model_params']
        self.train_cfg = cfg['train_params']
        self.multitask = self._cfg["multitask"]
        self.batch_size = self._cfg['train_params'].batch_size
        
        # TODO: maybe join both config files?
        
        self.device = torch_helpers.device
        self.model = TDMPC2WorldModel(
            env, **cfg
        ).to(self.device)

        self.optimizer = torch_helpers.optimizer_from_string(self.train_cfg.optimizer)(
            [{'params': self.model._encoder.parameters(), 'lr': self.train_cfg.lr*self.train_cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.multitask else []}],
            **self.train_cfg.optimizer_kwargs
        )

        self.pi_optim = torch_helpers.optimizer_from_string(self.train_cfg.optimizer)(
            self.model._pi.parameters(), **self.train_cfg.pi_optimizer_kwargs)
        
        self.model.eval()
        self.scale = RunningScale(cfg)
        #self.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in self.cfg.episode_lengths], device='cuda'
        ) if self.multitask else self._get_discount(self.cfg.episode_length)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.
        
        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, horizon : int):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.multitask else self.discount
        return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """		

        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.
        
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.
        
        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        
        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    # TODO: in eqn (3) in the paper, the expectation is computed over a sequence of shape [h, p, d] not just [p, d].
    #       Instead of sampling uniformly from the buffer, we probably have to sample trajectories instead.
    def update(self, obs, action, reward, task):
        """
        Main update function. Corresponds to one iteration of model learning.
        """
        batch_size = obs.shape[0]

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0

        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }

    @torch.no_grad
    def predict_n_steps(
        self, *, start_observations: np.ndarray, start_states: Sequence, policy: Controller, horizon, task
    ) -> Tuple[RolloutBuffer, np.ndarray]:

        all_latents_ = []
        all_actions_ = []

        states = start_states

        # preprocessing -> removes goals from playgroundwGoals env. Reduces obsdim from 48 to 40
        start_observations = self.env.obs_preproc(start_observations)

        # encode start_observations
        zs = self.model.encode(start_observations, task=task)
        all_latents_.append(zs)

        # iterate the forward model
        for i in range(horizon):
            # we query open loop policy, which just returns the next action.
            actions = policy.get_action(obs=zs, state=states)
            zs, states, rewards = self.model.predict(zs, actions, task=task)

            all_actions_.append(actions)
            all_latents_.append(zs)

        # from [h,p,d] to [p,h,d] p is for parallel and h is for horizon
        all_observations = torch.stack(all_latents_[:-1]).permute(1, 0, 2)
        all_next_observations = torch.stack(all_latents_[1:]).permute(1, 0, 2)
        all_actions =  torch.stack(all_actions_).permute(1, 0, 2)

        rollouts = SimpleRolloutBuffer(
            observations=all_observations,
            next_observations=all_next_observations,
            actions=all_actions,
        )

        return rollouts, None


    def get_state(self):
        return None

    def set_state(self, state):
        raise NotImplementedError

    def save(self, path):
        """
        Save state dict of the agent to filepath.
        
        Args:
            path (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            path (str or dict): Filepath or state dict to load.
        """
        state_dict = path if isinstance(path, dict) else torch.load(path)
        self.model.load_state_dict(state_dict["model"])

    def reset(self, observation):
        return None

    def got_actual_observation_and_env_state(self, *, observation, env_state=None, model_state=None):
        return None

    def rollout_generator(self, start_states, start_observations, horizon, policy, mode=None, task=None):
        raise NotImplementedError
        states = start_states
        obs = start_observations
        z = self.model.encode(obs, task=task)
        

    def rollout_field_names(self):
        raise NotImplementedError
    
    def predict(self, *, observations, states, actions, task=None) -> Tuple:
        raise NotImplementedError
        z = self.model.encode(observations, task=task)
        return self.model.next(z, actions, task=None)
    
    def train(
        self,
        rollout_buffer: RolloutBuffer,
        eval_buffer: Optional[RolloutBuffer] = None,
        maybe_update_normalizer=True
    ):

        num_rollouts = len(rollout_buffer)
        # TODO: reshape from [num_rollouts * episode_length, d] -> [num_rollouts, episode_length, d]
        observations = self.env.obs_preproc(rollout_buffer['observations']).reshape(num_rollouts, self.cfg.episode_length, -1)
        actions = rollout_buffer["actions"].reshape(num_rollouts, self.cfg.episode_length, -1)
        rewards = rollout_buffer["rewards"].reshape(num_rollouts, self.cfg.episode_length, -1)

        assert self.multitask == False, "Current Implementation doesn't support multitask"
        current_task = env_name_to_task(self.env)
        tasks = [current_task] * len(observations)
        training_statistics = {}

        # @Vera we should take a look at how exactly the batching works in the tdmpc implementation. 
        # Im not sure, but i think for tdmpc its important to keep rollouts separated. 
        # In Equation (3) of TDMPC2 they compute the loss as an expectation over a replay buffer and sample
        # trajectories of length `H` which is their horizon. So the last two dimensions of our tensors should be [h, d].
        # We should then probably take all observations and reshape from [num_rollouts, episode_length, d] into [b, h, d].
        # This would require some care, better discuss this more with marco and cansu... -> asked about this in slack

        """# simple batching
        batches_number = (len(observations) % self.batch_size != 0) + len(observations) // self.batch_size 
        
        
        for i in range(batches_number):

            # one batch should have size (h, d).
            obs = torch_helpers.to_tensor(observations[i*self.batch_size : (i+1) * self.batch_size]).to(torch_helpers.device)
            acts = torch_helpers.to_tensor(actions[i*self.batch_size : (i+1) * self.batch_size]).to(torch_helpers.device)
            r = torch_helpers.to_tensor(rewards[i*self.batch_size : (i+1) * self.batch_size]).to(torch_helpers.device)
            t = torch.Tensor(tasks[i*self.batch_size : (i+1) * self.batch_size]).to(torch_helpers.device)

            # TODO: reshape to [h, b, d]
            stats = self.update(obs, acts, r, t)
            training_statistics.update(stats)

        return training_statistics"""

        raise NotImplementedError

        
