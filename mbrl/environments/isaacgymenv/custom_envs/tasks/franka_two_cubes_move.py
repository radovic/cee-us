# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os


from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_conjugate
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaTwoCubesMove(VecTask):
    """
    This is a docstring!
    """
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_touch_scale": self.cfg["env"]["distTouchRewardScale"],
            "r_dist_target_scale": self.cfg["env"]["distTargetRewardScale"]
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: (cube_pose (7) + cube_velocity (3+3) + target_pos (3)) * 2 + eef_pose (7) + eef_velocity (3+3) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 47 if self.control_type == "osc" else 29
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        
        # Initial Cube values
        self._init_cube_state = None           # Initial state of cube for the current env
        self._init_cube2_state = None           
        self._init_target_state = None           # Initial state of target for the current env
        self._init_target2_state = None 
        self._cube_state = None                # Current state of cube for the current env
        self._cube2_state = None
        self._target_state = None                # Current state of target for the current env
        self._target2_state = None                # Current state of target 2 for the current env
        self._cube_id = None
        self._cube2_id = None                   
        self._target_id = None                   # Actor ID corresponding to target for a given env
        self._target2_id = None                   # Actor ID corresponding to target 2 for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cube_size = 0.050
        self.target_size = 0.070

        # Create cube asset
        cube_opts = gymapi.AssetOptions()
        cube_asset = self.gym.create_box(self.sim, *([self.cube_size] * 3), cube_opts)
        cube_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cube2 asset
        cube2_opts = gymapi.AssetOptions()
        cube2_asset = self.gym.create_box(self.sim, *([self.cube_size] * 3), cube2_opts)
        cube2_color = gymapi.Vec3(0.0, 0.0, 0.6)

        # Create target asset
        target_opts = gymapi.AssetOptions()
        target_opts.disable_gravity = True
        target_asset = self.gym.create_sphere(self.sim, self.target_size, target_opts)
        target_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create target2 asset
        target2_opts = gymapi.AssetOptions()
        target2_opts.disable_gravity = True
        target2_asset = self.gym.create_sphere(self.sim, self.target_size, target2_opts)
        target2_color = gymapi.Vec3(0.0, 0.0, 0.6)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cube_start_pose = gymapi.Transform()
        cube_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cube_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        cube2_start_pose = gymapi.Transform()
        cube2_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cube2_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        target_start_pose = gymapi.Transform()
        target_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        target_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        target2_start_pose = gymapi.Transform()
        target2_start_pose.p = gymapi.Vec3(1.0, 0.5, 0.0)
        target2_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 6     # + 6 is for for table, table stand, 2 cubes, 2 targets
        max_agg_shapes = num_franka_shapes + 6     # + 6 is for for table, table stand, 2 cubes, 2 targets

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 2, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 4, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cube_id = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 8, 0)
            self._cube2_id = self.gym.create_actor(env_ptr, cube2_asset, cube2_start_pose, "cube2", i, 16, 0)
            
            # Create targets
            self._target_id = self.gym.create_actor(env_ptr, 
                                                    target_asset, 
                                                    target_start_pose, 
                                                    "target", 
                                                    i, 
                                                    25, # binary: 11001, dont collide with franka arm and cube and second cube
                                                    0)
           
            self._target2_id = self.gym.create_actor(env_ptr,
                                                target2_asset,
                                                target2_start_pose,
                                                "target2",
                                                i,
                                                53, # binary: 110101, dont collide with franka arm and cube and second cube and target
                                                0)

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cube_id, 0, gymapi.MESH_VISUAL, cube_color)
            self.gym.set_rigid_body_color(env_ptr, self._cube2_id, 0, gymapi.MESH_VISUAL, cube2_color)
            self.gym.set_rigid_body_color(env_ptr, self._target_id, 0, gymapi.MESH_VISUAL, target_color)
            self.gym.set_rigid_body_color(env_ptr, self._target2_id, 0, gymapi.MESH_VISUAL, target2_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cube_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cube2_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_target_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_target2_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Cubes
            "cube_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cube_id, "box"),
            "cube2_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cube2_id, "box"),
            "target_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._target_id, "sphere"),
            "target2_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._target2_id, "sphere"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cube_state = self._root_state[:, self._cube_id, :]
        self._cube2_state = self._root_state[:, self._cube2_id, :]
        self._target_state = self._root_state[:, self._target_id, :]
        self._target2_state = self._root_state[:, self._target2_id, :]

        # Initialize states
        self.states.update({
            "cube_size": torch.ones_like(self._eef_state[:, 0]) * self.cube_size,
            "cube2_size": torch.ones_like(self._eef_state[:, 0]) * self.cube_size,
            "target_size": torch.ones_like(self._eef_state[:, 0]) * self.target_size,
            "target2_size": torch.ones_like(self._eef_state[:, 0]) * self.target_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        # global indices is a tensor of shape [num_envs, objs per env].
        num_objs_in_sim = 7
        self._global_indices = torch.arange(self.num_envs * num_objs_in_sim, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            
            # Cubes
            "cube_quat": self._cube_state[:, 3:7],
            "cube_pos": self._cube_state[:, :3],
            "cube_vel": self._cube_state[:, 7:],
            "cube2_quat": self._cube2_state[:, 3:7],
            "cube2_pos": self._cube2_state[:, :3],
            "cube2_vel": self._cube2_state[:, 7:],
            "cube_pos_relative": self._cube_state[:, :3] - self._eef_state[:, :3],
            "cube2_pos_relative": self._cube2_state[:, :3] - self._eef_state[:, :3],
            "target_quat": self._target_state[:, 3:7],
            "target2_quat": self._target2_state[:, 3:7],
            "target_pos": self._target_state[:, :3],
            "target2_pos": self._target2_state[:, :3],
            "cube_to_target_pos": self._target_state[:, :3] - self._cube_state[:, :3],
            "cube2_to_target2_pos": self._target2_state[:, :3] - self._cube2_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )
    
    def compute_reward_sas(self, observations, actions, next_observations):
        """
            Computation of the relative position of cube to target requires the next_observation to carry
            information about the target position. This is not possible with the current state of GNN, as we're forced
            to yield the position of the target in the observation, as every single observation HAS to be an object in the env
            for GNN to run. By yielding the relative position, we're including a non-object measurement into the observation
            which would crash the current GNN implementation.

            We suggest some form of masking out the target position in the observation,
            to still be able to return the positions for the target used to compute the relative position
            needed for reward computation. This solution would conform to the GNN implementation.
        """
        raise NotImplementedError
        # observation: cube_quat (4) + cube_pos (3), cube2 quat (4) + cube2_pos (3), + eef_pose (3) + q_gripper (2) = 19
        next_states = {
            'target_size': self.states['target_size'],
            'cube_size': self.states['cube_size'],
            'cube_pos': next_observations[..., 4:7], # 0-4 c1 rotation, 4-7 c1 pos, 7-11 c2 rotation, 11-14 c2 pos, 14-17 eef pos
            'cube2_pos': next_observations[..., 11:14],
            'cube_to_eef_pos': next_observations[..., 4:7] - next_observations[..., 14:17],
            'cube2_to_eef_pos': next_observations[..., 11:14] - next_observations[..., 14:17],
            'cube_to_target_pos': next_observations[..., 4:7], # TODO: chage indexing
            'cube2_to_target2_pos': next_observations[..., 18:21], # TODO: chage indexing
        }
        rewards, _ = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, next_states, self.reward_settings, self.max_episode_length
        )
        return rewards

    def compute_observations(self):
        self._refresh()
        # cube_pos (3) + cube_quat (4) + cube_vel (6) + cube_to_target_pos (3), cube2_pos (3) + cube2_quat (4) + cube2_vel (6) + cube2_to_target2_pos (3), eef_pose (3) + eef_quat (4) + eef_vel (6) + q_gripper (2) = 47
        obs = ["cube_pos", "cube_quat", "cube_vel", "cube_to_target_pos", "cube2_pos", "cube2_quat", "cube2_vel", "cube2_to_target2_pos", "eef_pos", "eef_quat", "eef_vel"] 
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}
        return self.obs_buf
    
    def compute_object_centric_observation(self, obs, agent_dim, object_dim, object_static_dim):
        if obs.ndim == 3:
            obs = obs.reshape(obs.shape[0], -1)
        elif obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        state_dict = {
            # TODO: add sth for target 2?
            "agent": obs[:, 32:],
            "objects_dyn": obs[:, :32].reshape(2, -1, 16),
            "objects_static": np.array([None] * 2)  # (obs[:, 4:7] + obs[:, 14:17]).reshape(1, -1, 3), # cube pos + cube to target pos = target pos
        }
        return state_dict
    
    # TODO: Improve this function to be more general
    def get_object_dims(self):
        agent_dim = 15
        object_dyn_dim = 16 
        object_stat_dim = 0
        nObj = 2 # two cubes
        return agent_dim, object_dyn_dim, object_stat_dim, nObj

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes initial position. Sample cube and target, making sure they dont intersect.
        self._reset_init_cube_state(env_ids=env_ids)

        # Write these new init states to the sim states
        self._cube_state[env_ids] = self._init_cube_state[env_ids]
        self._cube2_state[env_ids] = self._init_cube2_state[env_ids]
        self._target_state[env_ids] = self._init_target_state[env_ids]
        self._target2_state[env_ids] = self._init_target2_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten() 
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        
        # Update all actor states.
        # In our sim, the four last indices are the cube, cube2 and target and target2. Update them.
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -4:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, env_ids):
        """
            Note: It's important to set the rotation of all actors, otherwise weird stuff happens.
        """
        
        # TODO: make sure that sampling is collision free (currently targets could collide with each other)



        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device) # 4 for quat, 3 for pos, 6 for joint pos = 13
        sampled_cube2_state = torch.zeros(num_resets, 13, device=self.device)
        sampled_target_state = torch.zeros(num_resets, 13, device=self.device)
        sampled_target2_state = torch.zeros(num_resets, 13, device=self.device)

        cube_heights = self.states["cube_size"]
        target_heights = self.states["target_size"]

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists_cube_target = (self.states["cube_size"] + self.states["target_size"])[env_ids] * np.sqrt(2) / 2.0
        min_dists_cubes = (self.states["cube_size"] + self.states["cube_size"])[env_ids] * np.sqrt(2) / 2.0
        min_dists_targets = (self.states["target_size"] + self.states["target_size"])[env_ids] * np.sqrt(2) / 2.0


        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dist_cube_to_target = min_dists_cube_target * 2.0
        min_dist_target_to_target = min_dists_targets * 2.0
        min_dist_cube_to_cube = min_dists_cubes * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + torch.atleast_1d(cube_heights.squeeze(-1))[env_ids] / 2
        sampled_cube2_state[:, 2] = self._table_surface_pos[2] + torch.atleast_1d(cube_heights.squeeze(-1))[env_ids] / 2
        sampled_target_state[:, 2] = self._table_surface_pos[2] + torch.atleast_1d(target_heights.squeeze(-1))[env_ids] / 2
        sampled_target2_state[:, 2] = self._table_surface_pos[2] + torch.atleast_1d(target_heights.squeeze(-1))[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0
        sampled_cube2_state[:, 6] = 1.0
        sampled_target_state[:, 6] = 1.0
        sampled_target2_state[:, 6] = 1.0

        # first sample the cubes position. Uniform distr. [-1, 1] \times [-1, 1] around centered_cube_xy_state
        sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                            2.0 * self.start_position_noise * (
                                                    torch.rand(num_resets, 2, device=self.device) - 0.5)
        
        # collision free sampling of the target. 
        success = False

        # Indexes corresponding to envs we're still actively sampling for
        active_idx = torch.arange(num_resets, device=self.device)
        num_active_idx = len(active_idx)

        for i in range(200): 
            # Sample x y values. Uniform distr. [-1, 1] \times [-1, 1] around centered_cube_xy_state
            sampled_target_state[active_idx, :2] = centered_cube_xy_state + \
                                                    2.0 * self.start_position_noise * (
                                                            torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
            
           
            sampled_cube2_state[active_idx, :2] = centered_cube_xy_state + \
                                                    2.0 * self.start_position_noise * (
                                                            torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
            
            sampled_target2_state[active_idx, :2] = centered_cube_xy_state + \
                                                    2.0 * self.start_position_noise * (
                                                            torch.rand_like(sampled_cube2_state[active_idx, :2]) - 0.5)
            
            # Check if sampled values are valid
            cube_to_target_dist = torch.linalg.norm(sampled_cube_state[:, :2] - sampled_target_state[:, :2], dim=-1)
            cube2_to_target2_dist = torch.linalg.norm(sampled_cube2_state[:, :2] - sampled_target2_state[:, :2], dim=-1)
            cube_to_cube2_dist = torch.linalg.norm(sampled_cube_state[:, :2] - sampled_cube2_state[:, :2], dim=-1)
            target_to_target2_dist = torch.linalg.norm(sampled_target_state[:, :2] - sampled_target2_state[:, :2], dim=-1)
            active_idx = torch.nonzero((cube_to_cube2_dist < min_dist_cube_to_cube) | (cube2_to_target2_dist < min_dist_cube_to_target) | (cube_to_target_dist < min_dist_cube_to_target) | (target_to_target2_dist < min_dist_target_to_target), as_tuple=True)[0]
            num_active_idx = len(active_idx)

            # If active idx is empty, then all sampling is valid
            if num_active_idx == 0:
                success = True
                break
        
        # Make sure we succeeded at sampling
        assert success, "Sampling cube locations was unsuccessful!"
    
        # Sample rotation value for cube
        if self.start_rotation_noise > 0:
            aa_rot_cube = torch.zeros(num_resets, 3, device=self.device)
            aa_rot_cube[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)

            aa_rot_cube2 = torch.zeros(num_resets, 3, device=self.device)
            aa_rot_cube2[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            
            aa_rot_target = torch.zeros(num_resets, 3, device=self.device)
            aa_rot_target[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            
            aa_rot_target2 = torch.zeros(num_resets, 3, device=self.device)
            aa_rot_target2[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)

            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot_cube), sampled_cube_state[:, 3:7])
            sampled_cube2_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot_cube2), sampled_cube2_state[:, 3:7])
            sampled_target_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot_target), sampled_target_state[:, 3:7])
            sampled_target2_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot_target2), sampled_target2_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        self._init_cube_state[env_ids, :] = sampled_cube_state
        self._init_cube2_state[env_ids, :] = sampled_cube2_state
        self._init_target_state[env_ids, :] = sampled_target_state
        self._init_target2_state[env_ids, :] = sampled_target2_state


    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def _compute_success(self):
        success = torch.norm(self.obs_buf[:, 14:17], dim=-1) < self.target_size
        self.extras['success'] = torch.atleast_1d(success)
        return success

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        '''
        if self.control_type == "osc":
            # TEST: Trying to fix orientation
            current = self.states["eef_quat"]
            cc = quat_conjugate(current)
            desired = torch.ones_like(current)
            desired[:, [2, 3]] = 0
            desired /= torch.norm(desired)
            q_r = quat_mul(desired, cc)
            orientation_offset = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1) * 100
            orientation_offset = torch.zeros_like(u_arm)
            u_arm = torch.cat([u_arm, orientation_offset], dim=-1)
        '''    
        
        u_arm = u_arm * self.cmd_limit / self.action_scale

        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self._compute_success()

        # debug viz
        if self.viewer and self.debug_viz:
            raise NotImplementedError
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cube_pos = self.states["cube_pos"]
            cube_rot = self.states["cube_quat"]
            cube2_pos = self.states["cube2_pos"]
            cube2_rot = self.states["cube2_quat"]
            target_pos = self.states["target_pos"]
            target_rot = self.states["target_quat"]
            target2_pos = self.states["target2_pos"]
            target2_rot = self.states["target2_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cube_pos, cube2_pos, target_pos, target2_pos), (eef_rot, cube_rot, cube2_rot, target_rot, target2_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], p0[3], p0[4], px[0], px[1], px[2], px[3], px[4]], [0.85, 0.1, 0.1]) #TODO: maybe add some more elems in array
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], p0[3], p0[4], py[0], py[1], py[2], py[3], py[4]], [0.1, 0.85, 0.1]) # same
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], p0[3], p0[4], pz[0], pz[1], pz[2], pz[3], pz[4]], [0.1, 0.1, 0.85]) # same

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    target_height = states["target_size"] + states["cube_size"] / 2.0
    cube_size = states["cube_size"]
    target_size = states["target_size"]

    # distance from hand to the cube
    d1 = torch.norm(states["cube_pos_relative"], dim=-1)
    d2 = torch.norm(states["cube2_pos_relative"], dim=-1)
    # d_lf = torch.norm(states["cube_pos"] - states["eef_lf_pos"], dim=-1)
    # d_rf = torch.norm(states["cube_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * d1) - torch.tanh(10.0 * d2)

    # distance from the cube to the target
    '''
    offset = torch.zeros_like(states["cube_to_target_pos"])
    offset2 = torch.zeros_like(states["cube2_to_target2_pos"])
    if offset.ndim > 2: offset.transpose(1, 2)[..., 2] = (cube_size + target_size) / 2
    else: offset[..., 2] = (cube_size + target_size) / 2
    if offset2.ndim > 2: offset2.transpose(1, 2)[..., 2] = (cube_size + target_size) / 2
    else: offset2[..., 2] = (cube_size + target_size) / 2
    # offset[..., 2] = (cube_size + target_size) / 2
    '''
    d_cube_target = torch.norm(states["cube_to_target_pos"], dim=-1)
    d_cube2_target2 = torch.norm(states["cube2_to_target2_pos"], dim=-1)
    dist_target_reward = 1 - torch.tanh(10.0 * d_cube_target) - torch.tanh(10.0 * d_cube2_target2)

    reward_settings['r_dist_target_scale'] = 1.0
    reward_settings['r_dist_touch_scale'] = 0.1

    rewards = reward_settings['r_dist_target_scale'] * dist_target_reward +\
        dist_reward * reward_settings['r_dist_touch_scale']

    # Compute resets
    reset_buf = torch.where(progress_buf >= (max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf