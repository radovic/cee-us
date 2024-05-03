import numpy as np
import torch
import torch.nn as nn

from mbrl import torch_helpers
from mbrl.controllers.abstract_controller import TrainableController
from mbrl.controllers.mpc_torch import TorchMpcICem
from mbrl.models.utils import build_mlp
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl.torch_helpers import TrainingIterator


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class RMS(object):
    """
    Running Mean and Standard Deviation (RMS) Calculator.

    This class calculates the running mean and standard deviation of a given 
    input tensor.
    
    Parameters:
        device: The device (e.g. 'cude:0') where the RMS calculator will be run
        epsilon: A small value added to the numerator of the running mean calculation
            to prevent division by zero. Defaults to 1e-4.
        shape: The shape of the internal buffers for mean and standard deviation,
            which should match the expected input tensor shape. Defaults to (1,).

    Methods:
        __call__(x): Calculates the running mean and standard deviation of a 
            given input tensor `x`.
            Returns the updated running mean and standard deviation.

    Notes:
        - This implementation uses the Welford's algorithm 
            (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
            for online calculation of the mean and variance.
        - The RMS calculator maintains its own internal state, which is updated 
            each time it processes an input tensor.

    """

    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + torch.square(delta) * self.n * bs / (self.n + bs)) / (
            self.n + bs
        )

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class RND(nn.Module):
    """
    Random Network Distillation (RND) model.

    This module implements the Random Network Distillation (RND) algorithm, which
    learns to predict the behavior of an agent in a given environment. The
    predictions are based on observations from the environment and are used to
    regularize the agent's policy updates.

    Parameters:
        - `model_params`: A dictionary of hyperparameters for the RND model.
            This includes parameters such as the observation dimension, the 
            number
            of layers in the predictor and target networks, etc.
        - `clip_val` (optional): The value to clip observations to. Defaults to 
            5.0.

    Attributes:
        - `predictor`: A neural network that predicts the behavior of the agent.
        - `target`: A neural network that serves as a teacher for the predictor.
                    `requires_grad` is set to False for this Module. 
        - `obs_normalizer`: A normalizer for the observations.


    Methods:
        - `_parse_model_params` : Parses the model parameters and sets the
            attributes accordingly.
        - `forward` : Computes the prediction error between the predictor's
            predictions and the target network's outputs.
    """
    def __init__(self, model_params, clip_val=5.0):

        super().__init__()
        self._parse_model_params(**model_params)
        self.clip_val = clip_val

        if self.rnd_network_type == "mlp":

            if self.obs_dim != self.agent_dim + self.nObj * (self.object_dyn_dim + self.object_stat_dim):
                print("Running RND in a non-object-centric environment!")

            self.obs_normalizer = torch_helpers.Normalizer(self.obs_dim, eps=1e-6)

            self.predictor = build_mlp(
                input_dim=self.obs_dim,
                output_dim=self.rnd_rep_dim,
                size=self.rnd_hidden_dim,
                num_layers=self.num_layers_predictor,  # 1
                activation="relu",
                layer_norm=False,
            )

            self.target = build_mlp(
                input_dim=self.obs_dim,
                output_dim=self.rnd_rep_dim,
                size=self.rnd_hidden_dim,
                num_layers=self.num_layers_target,  # 1
                activation="relu",
                layer_norm=False,
            )

        else:
            raise NotImplementedError

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(weight_init)
        self.to(torch_helpers.device)

    def _parse_model_params(
        self,
        *,
        agent_dim,
        object_dyn_dim,
        object_stat_dim,
        nObj,
        obs_dim,
        num_layers_target,
        num_layers_predictor,
        rnd_hidden_dim,
        rnd_rep_dim,
        rnd_network_type="mlp",
    ):
        # Necessary to get the observation dimension
        self.agent_dim = agent_dim
        self.object_dyn_dim = object_dyn_dim
        self.object_stat_dim = object_stat_dim
        self.nObj = nObj
        self.obs_dim = obs_dim

        self.num_layers_target = num_layers_target
        self.num_layers_predictor = num_layers_predictor
        self.rnd_hidden_dim = rnd_hidden_dim
        self.rnd_rep_dim = rnd_rep_dim
        self.rnd_network_type = rnd_network_type

    def forward(self, obs):
        obs = self.obs_normalizer.normalize(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(dim=-1, keepdim=True)
        return prediction_error


class TorchRNDMpcICem(TrainableController, TorchMpcICem):
    def __init__(self, *, model_params, train_params, extrinsic_reward=False, extrinsic_reward_scale=1.0, **kwargs):

        super().__init__(**kwargs)
        self._w_extrinsic_reward = extrinsic_reward
        self._maybe_extrinsic_reward_scale = extrinsic_reward_scale
        self.update_model_param_env_depend(model_params)
        self._parse_train_params(**train_params)

        self.rnd = RND(model_params)
        self.intrinsic_reward_rms = RMS(device=torch_helpers.device)
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.rnd_lr)
        self.rnd.train()

    def _parse_train_params(
        self,
        *,
        learning_rate,
        batch_size,
        rnd_epochs=True,
        rnd_num_epochs_or_its=20,
    ):
        self.rnd_lr = learning_rate
        self.batch_size = batch_size
        if rnd_epochs:
            self.epochs = rnd_num_epochs_or_its
            self.iterations = 0
        else:
            self.iterations = rnd_num_epochs_or_its
            self.epochs = 0

    def update_model_param_env_depend(self, model_params):
        return model_params.update(
            agent_dim=self.env.agent_dim,
            object_dyn_dim=self.env.object_dyn_dim,
            object_stat_dim=self.env.object_stat_dim,
            nObj=self.env.nObj,
            obs_dim=self.env.observation_space_size_preproc,
        )

    def update_rnd(self, obs):
        self.rnd.train()

        prediction_error = self.rnd(obs)

        loss = prediction_error.mean()

        # gradients flow only to the predictor, as requires_grad is
        # set False for the target net.
        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()

        return {"rnd_loss": loss.item()}

    @torch.no_grad()
    def _update_normalizer(self, rollout_buffer: RolloutBuffer):
        latest_observations = rollout_buffer.latest_rollouts["observations"]
        self.rnd.obs_normalizer.update(self.env.obs_preproc(latest_observations.reshape(-1, latest_observations.shape[-1])))

    def train_rnd(self, rollout_buffer: RolloutBuffer):
        self._update_normalizer(rollout_buffer)
        observations = rollout_buffer["observations"]

        iterator_train = TrainingIterator(
            data_dict=dict(
                inputs=self.env.obs_preproc(observations),
            )
        )

        iterator = None
        if self.epochs:
            iterator = iterator_train.get_epoch_iterator(self.batch_size, self.epochs)
            epoch_length = np.ceil(iterator_train.array["inputs"].shape[0] / self.batch_size)
        elif self.iterations:
            iterator = iterator_train.get_basic_iterator(self.batch_size, self.iterations)
            epoch_length = 1

        train_loss_accum = 0.0
        for i, batch in enumerate(iterator()):
            rnd_loss_dict = self.update_rnd(batch["inputs"])
            train_loss_accum += rnd_loss_dict["rnd_loss"]
            if (i + 1) % epoch_length == 0 or i == 0:
                rnd_loss = train_loss_accum / min(epoch_length, i + 1)
                self.logger.log(rnd_loss, key="train/rnd_loss")
                train_loss_accum = 0.0
        return {
            "epoch_rnd_loss": rnd_loss,
        }

    def train(self):
        pass

    @torch.no_grad()
    def compute_intr_reward(self, obs):
        self.rnd.eval()
        prediction_error = self.rnd(self.env.obs_preproc(obs))
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        return prediction_error / (torch.sqrt(intr_reward_var) + 1e-8)

    @torch.no_grad()
    def preallocate_memory(self):
        """
        Preallocate memory for distribution parameters in addition
        """
        super().preallocate_memory()

        self._rnd_bonus_per_path = torch.empty(
            (
                self.num_sim_traj,
                self.horizon,
            ),
            device=torch_helpers.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    def _model_epistemic_costs(self, rollout_buffer: RolloutBuffer):

        mean_next_obs = rollout_buffer.as_array("next_observations")  # shape: [nenvs, p,h,obs_dim]
        rnds_of_samples_ = self.compute_intr_reward(mean_next_obs.reshape(-1, self.env.observation_space.shape[0]))
        self._rnd_bonus_per_path = rnds_of_samples_.view(mean_next_obs.shape[0], mean_next_obs.shape[1], self.horizon)  # [nenvs, p, h]

    @torch.no_grad()
    def trajectory_cost_fn(self, cost_fn, rollout_buffer: RolloutBuffer, out: torch.Tensor):
        if self.use_env_reward:
            raise NotImplementedError()
            # costs_path shape: [p,h]

        self._model_epistemic_costs(rollout_buffer)

        costs_path = -self._rnd_bonus_per_path

        if self._w_extrinsic_reward:
            env_cost = cost_fn(
                rollout_buffer.as_array("observations"),
                rollout_buffer.as_array("actions"),
                rollout_buffer.as_array("next_observations"),
            )  # shape: [p,h]
            costs_path += self._maybe_extrinsic_reward_scale * env_cost

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

    # Save model parameters
    def save(self, path):
        torch.save(
            {
                "rnd_target": self.rnd.target.state_dict(),
                "rnd_predictor": self.rnd.predictor.state_dict(),
                "obs_normalizer": self.rnd.obs_normalizer.state_dict(),
                "optimizer": self.rnd_opt.state_dict(),
            },
            path,
        )

    def load(self, path):
        state_dicts = torch.load(path)
        self.rnd.target.load_state_dict(state_dicts["rnd_target"])
        self.rnd.predictor.load_state_dict(state_dicts["rnd_predictor"])
        self.rnd.obs_normalizer.load_state_dict(state_dicts["obs_normalizer"])
        self.rnd_opt.load_state_dict(state_dicts["optimizer"])
