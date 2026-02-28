# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from ..lite3.lite3_config import Lite3RoughCfg
from ..x30.x30_config import X30RoughCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # 保存配置参数和仿真参数
        self.cfg = cfg  # 环境配置参数
        self.sim_params = sim_params  # 仿真物理参数
        self.height_samples = None  # 高度采样点（用于地形高度测量）
        self.debug_viz = False  # 调试可视化开关
        self.init_done = False  # 初始化完成标志
        
        # 解析配置文件，设置各种参数和奖励函数
        self._parse_cfg(self.cfg)
        
        # 调用父类初始化方法，创建基础环境
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        # 设置观测维度参数
        self.num_one_step_obs = self.cfg.env.num_one_step_observations  # 单步观测维度
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs  # 单步特权观测维度
        self.history_length = int(self.num_obs / self.num_one_step_obs)  # 历史观测长度（用于历史信息）

        # 如果不是无头模式，设置相机视角
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        # 初始化各种缓冲区（状态、奖励等）
        self._init_buffers()
        
        # 准备奖励函数
        self._prepare_reward_function()
        
        # 标记初始化完成
        self.init_done = True

    def step(self, actions):
        # 执行动作，进行仿真，调用后处理步骤
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        # 物理仿真后的处理步骤，包括奖励计算、观测更新等
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
        # 计算接触状态和接触过滤（用于奖励函数）
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.  # [num_envs, 4]
        self.contact_filt = torch.logical_or(contact, self.last_contacts)  # [num_envs, 4]
        self.last_contacts = contact

        # 记录当前高度到缓冲区
        current_height = self._get_base_heights()  # 使用相对高度
        self.episode_height_buffer += current_height
        self.episode_height_count += 1
        
        # 记录当前髋关节角度到缓冲区
        current_hip_angle = self._get_hip_angles()  # 获取髋关节角度
        self.episode_hip_angle_buffer += current_hip_angle
        self.episode_hip_angle_count += 1
        
        # 记录当前大腿关节角度到缓冲区
        current_thigh_angle = self._get_thigh_angles()  # 获取大腿关节角度
        self.episode_thigh_angle_buffer += current_thigh_angle
        self.episode_thigh_angle_count += 1
        
        # 记录当前膝关节角度到缓冲区
        current_knee_angle = self._get_knee_angles()  # 获取膝关节角度
        self.episode_knee_angle_buffer += current_knee_angle
        self.episode_knee_angle_count += 1

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        # 检查回合是否应该终止（摔倒、超时等）
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        # 重置指定环境的机器人状态
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        


        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if len(env_ids) > 0:
            # 计算episode运行过程中的平均高度
            # 确保只在有数据的环境上计算
            valid_envs = self.episode_height_count[env_ids] > 0
            if torch.any(valid_envs):
                valid_env_ids = env_ids[valid_envs]
                avg_height = self.episode_height_buffer[valid_env_ids] / self.episode_height_count[valid_env_ids]
                self.extras["episode"]["current_height"] = torch.mean(avg_height)
            else:
                # 如果没有有效数据，使用当前高度
                current_height = self._get_base_heights()[env_ids]
                self.extras["episode"]["current_height"] = torch.mean(current_height)
            # 重置高度缓冲区
            self.episode_height_buffer[env_ids] = 0.
            self.episode_height_count[env_ids] = 0.
            
            # 添加髋关节角度信息到episode日志
            valid_envs = self.episode_hip_angle_count[env_ids] > 0
            if torch.any(valid_envs):
                valid_env_ids = env_ids[valid_envs]
                avg_hip_angle = self.episode_hip_angle_buffer[valid_env_ids] / self.episode_hip_angle_count[valid_env_ids]
                self.extras["episode"]["current_hip_angle"] = torch.mean(avg_hip_angle)
            else:
                # 如果没有有效数据，使用当前髋关节角度
                current_hip_angle = self._get_hip_angles()[env_ids]
                self.extras["episode"]["current_hip_angle"] = torch.mean(current_hip_angle)
            # 重置髋关节角度缓冲区
            self.episode_hip_angle_buffer[env_ids] = 0.
            self.episode_hip_angle_count[env_ids] = 0.
            
            # 添加大腿关节角度信息到episode日志
            valid_envs = self.episode_thigh_angle_count[env_ids] > 0
            if torch.any(valid_envs):
                valid_env_ids = env_ids[valid_envs]
                avg_thigh_angle = self.episode_thigh_angle_buffer[valid_env_ids] / self.episode_thigh_angle_count[valid_env_ids]
                self.extras["episode"]["current_thigh_angle"] = torch.mean(avg_thigh_angle)
            else:
                # 如果没有有效数据，使用当前大腿关节角度
                current_thigh_angle = self._get_thigh_angles()[env_ids]
                self.extras["episode"]["current_thigh_angle"] = torch.mean(current_thigh_angle)
            # 重置大腿关节角度缓冲区
            self.episode_thigh_angle_buffer[env_ids] = 0.
            self.episode_thigh_angle_count[env_ids] = 0.
            
            # 添加膝关节角度信息到episode日志
            valid_envs = self.episode_knee_angle_count[env_ids] > 0
            if torch.any(valid_envs):
                valid_env_ids = env_ids[valid_envs]
                avg_knee_angle = self.episode_knee_angle_buffer[valid_env_ids] / self.episode_knee_angle_count[valid_env_ids]
                self.extras["episode"]["current_knee_angle"] = torch.mean(avg_knee_angle)
            else:
                # 如果没有有效数据，使用当前膝关节角度
                current_knee_angle = self._get_knee_angles()[env_ids]
                self.extras["episode"]["current_knee_angle"] = torch.mean(current_knee_angle)
            # 重置膝关节角度缓冲区
            self.episode_knee_angle_buffer[env_ids] = 0.
            self.episode_knee_angle_count[env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        # 计算奖励函数，汇总所有奖励项
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        # 计算观测值，包括当前状态和历史信息
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale, # 运动指令 （xy线速度，偏航角）*缩放因子 [3维]
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 基座角速度 （位置，速度，姿态）*缩放因子 [3维]
                                    self.projected_gravity, # 重力投影 [3维]
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 关节位置 （关节位置-默认位置）*缩放因子 [12维]
                                    self.dof_vel * self.obs_scales.dof_vel, # 关节速度 关节速度*缩放因子 [12维]
                                    self.actions # 动作 [12维]
                                    ),dim=-1)
        #总共：3+3+3+12+12+12=45维
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]  # 9 = 3(命令) + 3(角速度) + 3(重力)

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        #地形高度信息187维
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        if self.cfg.env.use_history_obs:
            # 将当前观测和历史观测拼接起来
            self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], #当前观测47维
            self.obs_buf[:, :-self.num_one_step_obs]), #历史5步的观测
            dim=-1)
            # 特权观测
            self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)
        else:
            # 只使用当前观测，不拼接历史信息
            self.obs_buf = current_obs[:, :self.num_one_step_obs]  # 只使用当前观测47维
            self.privileged_obs_buf = current_obs[:, :self.num_one_step_privileged_obs]# 特权观测（也只使用单步观测）

    def get_current_obs(self):
        # 获取当前时刻的观测值
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale, # 运动指令 [3维]
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 基座角速度 [3维]
                                    self.projected_gravity, # 重力投影 [3维]
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 关节位置 [12维]
                                    self.dof_vel * self.obs_scales.dof_vel, # 关节速度 [12维]
                                    self.actions # 动作 [12维]
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]  # 9 = 3(命令) + 3(角速度) + 3(重力)

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return current_obs
        
    def compute_termination_observations(self, env_ids):
        # 计算终止时的观测值，用于训练
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale, # 运动指令 [3维]
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 基座角速度 [3维]
                                    self.projected_gravity, # 重力投影 [3维]
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 关节位置 [12维]
                                    self.dof_vel * self.obs_scales.dof_vel, # 关节速度 [12维]
                                    self.actions # 动作 [12维]
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]  # 9 = 3(命令) + 3(角速度) + 3(重力)

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        # 创建仿真环境，包括地形和机器人
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        # 设置相机位置和视角
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        # 处理刚体形状属性，如摩擦系数等
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        # 刷新指定环境的刚体形状属性
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        # 处理关节属性，如阻尼、刚度等
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # 处理刚体属性，如质量、惯性等
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        # 物理仿真后的回调函数，用于子类扩展
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        # 选择需要重新采样命令的环境（当episode长度达到重采样时间的整数倍时）
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # 为选中的环境重新生成运动命令
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            # 计算机器人当前的朝向向量
            forward = quat_apply(self.base_quat, self.forward_vec)
            # 计算当前航向角
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 根据目标航向角和当前航向角的差值计算角速度命令
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)

        # 如果启用地形高度测量，获取当前地形高度信息
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # 如果启用随机推动且达到推动间隔，随机推动机器人
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        # 如果启用扰动且达到扰动间隔，对机器人施加随机扰动
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        # 重新采样运动指令（速度、方向等）
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # 为指定环境生成随机的X方向线速度命令（前进/后退）
        self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        # 为指定环境生成随机的Y方向线速度命令（左右移动）
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            # 如果使用航向命令模式，生成目标航向角
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            # 否则生成偏航角速度命令（原地旋转）
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # 不再需要风格维度采样（只有一种风格）

        # 选择前20%的环境作为高速环境
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

        # 为高速环境生成更高速度的X方向命令
        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)

        # 当X方向速度较大时，将Y方向命令设为0（避免斜向运动）
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # 将总速度小于0.2的命令设为0（避免微小抖动）
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        # 根据动作计算关节力矩
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *=self.cfg.control.hip_reduction
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        # 重置指定环境的关节状态
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        根据风格命令选择不同的初始关节角度：
        - crouch: 使用趴下姿态
        - normal: 使用通用姿态

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # 使用默认关节角度
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        # 重置指定环境的基座状态（位置、速度等）
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # 初始化默认状态
        init_states = self.base_init_state.unsqueeze(0).repeat(len(env_ids), 1)
        
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = init_states
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = init_states
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        # 随机推动机器人，增加训练鲁棒性
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        # 对机器人施加扰动，模拟外部干扰
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        # 更新地形课程学习，逐步增加难度
        """ 实现游戏式的地形课程学习机制
        根据机器人的行走表现动态调整地形难度：
        - 表现好的机器人（走得足够远）升级到更难的地形
        - 表现差的机器人（走得不够远）降级到更简单的地形
        - 达到最高等级的机器人会被随机分配到某个等级继续训练

        Args:
            env_ids (List[int]): 需要重置的环境ID列表
        """
        # 如果环境还未初始化完成，不进行课程学习更新
        if not self.init_done:
            # 初始重置时不改变地形等级
            return
        
        # 计算每个环境中的机器人从起始位置到当前位置的行走距离（x-y平面）
        # root_states[env_ids, :2] 是当前位置的x,y坐标
        # env_origins[env_ids, :2] 是起始位置的x,y坐标
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        
        # 判断哪些机器人应该升级：行走距离超过地形长度的一半
        # 这些机器人表现良好，可以挑战更难的地形
        move_up = distance > self.terrain.env_length / 2
        
        # 判断哪些机器人应该降级：行走距离小于预期距离的一半
        # 预期距离 = 指令速度的模长 * 最大回合时长 * 0.5
        # 这些机器人表现不佳，需要回到更简单的地形练习
        # ~move_up 确保不会同时升级和降级
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        
        # 更新地形等级：升级的+1，降级的-1
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        
        # 处理达到最高等级的情况：随机分配到某个等级继续训练
        # 这样可以避免机器人只在最高等级训练，保持训练的多样性
        # 同时确保等级不会小于0（最低等级为0）
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (最低等级为0)
        
        # 根据新的地形等级和地形类型，更新环境的起始位置
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        # 更新指令课程学习，逐步增加运动难度
        """ 实现指令课程学习机制
        根据机器人的速度跟踪表现动态调整速度指令范围：
        - 将环境分为低速组（后80%）和高速组（前20%）
        - 如果两组的跟踪奖励都超过最大值的80%，则扩大速度指令范围
        - 这样可以逐步增加训练难度，让机器人学习更快的运动

        Args:
            env_ids (List[int]): 需要重置的环境ID列表
        """
        # 将环境分为两组：
        # - 低速组：环境ID大于总环境数的20%（后80%的环境）
        # - 高速组：环境ID小于总环境数的20%（前20%的环境）
        # 这样可以在不同速度范围内同时训练，提高训练效率
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        
        # 提取实际的环境ID索引
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        
        # 如果低速组和高速组的跟踪奖励都超过最大值的80%，则扩大速度指令范围
        # 跟踪奖励 = 回合累计奖励 / 最大回合长度
        # 当两组都表现良好时，说明机器人已经掌握了当前速度范围，可以挑战更快的速度
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            # 扩大x方向线速度指令范围：最小值减少0.2，最大值增加0.2
            # 使用clip确保不会超过配置的最大课程学习范围
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        # 获取噪声缩放向量，用于观测噪声
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)  # 9 = 3(命令) + 3(角速度) + 3(重力)
        else:
            noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)  # 9 = 3(命令) + 3(角速度) + 3(重力)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands (lin_vel_x, lin_vel_y, ang_vel_yaw)
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # 基座角速度
        noise_vec[6:9] = noise_scales.gravity * noise_level  # 重力投影
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 关节位置
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 关节速度
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0. # previous actions (动作不需要噪声)
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        #noise_vec[232:] = 0
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        # 初始化各种缓冲区，存储状态、观测等数据
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])

        # 添加高度缓冲区，用于记录episode运行过程中的高度
        self.episode_height_buffer = torch.zeros(self.num_envs, device=self.device)  # 累积高度
        self.episode_height_count = torch.zeros(self.num_envs, device=self.device)   # 高度记录次数
        
        # 添加髋关节角度缓冲区，用于记录episode运行过程中的髋关节角度
        self.episode_hip_angle_buffer = torch.zeros(self.num_envs, device=self.device)  # 累积髋关节角度
        self.episode_hip_angle_count = torch.zeros(self.num_envs, device=self.device)   # 髋关节角度记录次数
        
        # 添加大腿关节角度缓冲区，用于记录episode运行过程中的大腿关节角度
        self.episode_thigh_angle_buffer = torch.zeros(self.num_envs, device=self.device)  # 累积大腿关节角度
        self.episode_thigh_angle_count = torch.zeros(self.num_envs, device=self.device)   # 大腿关节角度记录次数
        
        # 添加膝关节角度缓冲区，用于记录episode运行过程中的膝关节角度
        self.episode_knee_angle_buffer = torch.zeros(self.num_envs, device=self.device)  # 累积膝关节角度
        self.episode_knee_angle_count = torch.zeros(self.num_envs, device=self.device)   # 膝关节角度记录次数
        

        
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化默认关节角度
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        # 初始化趴下时的关节角度（如果配置中存在）
        # 初始化基座高度目标
        if hasattr(self.cfg.rewards, 'base_height_target_normal'):
            self.base_height_target_normal = self.cfg.rewards.base_height_target_normal
        else:
            self.base_height_target_normal = self.cfg.rewards.base_height_target
        
        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


    def _prepare_reward_function(self):
        # 准备奖励函数，设置各种奖励项的权重
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        # 创建平地地形
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        # 创建高度场地形（如起伏地形）
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        # 创建三角网格地形（如楼梯、障碍物）
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        # 创建多个仿真环境实例
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # 获取URDF文件的完整路径，并替换LEGGED_GYM_ROOT_DIR变量
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        # 提取URDF文件所在的目录路径（用于mesh文件查找）
        asset_root = os.path.dirname(asset_path)
        # 提取URDF文件名（不包含路径）
        asset_file = os.path.basename(asset_path)

        # 创建Isaac Gym的Asset选项对象，用于配置机器人加载参数
        asset_options = gymapi.AssetOptions()
        # 设置关节驱动模式：0=无驱动，1=位置控制，2=速度控制，3=力矩控制
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # 是否合并固定关节：True表示将固定连接的刚体合并为一个刚体，提高仿真效率
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        # 是否将圆柱体碰撞体替换为胶囊体：胶囊体碰撞检测更稳定快速
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # 是否翻转视觉附件：某些mesh文件需要从Y轴向上翻转到Z轴向上（Isaac Gym使用Z轴向上）
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # 是否固定基座链接：True表示机器人基座不会移动
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        # 设置刚体密度：用于计算质量（如果URDF中没有指定质量）
        asset_options.density = self.cfg.asset.density
        # 设置角阻尼：模拟空气阻力等阻尼效应
        asset_options.angular_damping = self.cfg.asset.angular_damping
        # 设置线阻尼：模拟空气阻力等阻尼效应
        asset_options.linear_damping = self.cfg.asset.linear_damping
        # 设置最大角速度：限制刚体的最大旋转速度
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # 设置最大线速度：限制刚体的最大移动速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # 设置关节电枢：模拟电机惯性
        asset_options.armature = self.cfg.asset.armature
        # 设置厚度：用于薄壁物体的碰撞检测
        asset_options.thickness = self.cfg.asset.thickness
        # 是否禁用重力：True表示该物体不受重力影响
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 加载机器人Asset：从URDF文件加载机器人模型，包括所有mesh文件和物理属性
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # 获取机器人的自由度数量（关节数量）
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        # 获取机器人的刚体数量（链接数量）
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        # 获取关节属性：包括位置限制、速度限制、力矩限制等
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # 获取刚体形状属性：包括碰撞体形状、摩擦系数、恢复系数等
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 从Asset中获取所有刚体名称列表
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # 从Asset中获取所有关节名称列表
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # 计算刚体数量（重新赋值确保一致性）
        self.num_bodies = len(body_names)
        # 计算关节数量
        self.num_dofs = len(self.dof_names)
        # 筛选出足部刚体名称：查找包含foot_name的刚体名称
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # 初始化被惩罚接触的刚体名称列表
        penalized_contact_names = []
        # 遍历配置中指定的被惩罚接触的刚体类型，找到对应的刚体名称
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        # 初始化终止接触的刚体名称列表
        termination_contact_names = []
        # 遍历配置中指定的终止接触的刚体类型，找到对应的刚体名称
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        # 创建默认刚体质量张量：用于存储每个刚体的质量
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        # 组合基座初始状态：位置+旋转+线速度+角速度
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        # 将基座初始状态转换为PyTorch张量
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        
        # 创建Isaac Gym的变换对象，用于设置机器人初始位姿
        start_pose = gymapi.Transform()
        # 设置初始位置：使用基座初始状态的前3个元素（x, y, z坐标）
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 获取所有环境的起始位置（地形相关）
        self._get_env_origins()
        # 设置环境边界框的下界（用于环境创建）
        env_lower = gymapi.Vec3(0., 0., 0.)
        # 设置环境边界框的上界（用于环境创建）
        env_upper = gymapi.Vec3(0., 0., 0.)
        # 初始化Actor句柄列表：存储每个环境中机器人的句柄
        self.actor_handles = []
        # 初始化环境句柄列表：存储每个环境的句柄
        self.envs = []
        
        # 创建载荷质量张量：用于域随机化中的载荷质量变化
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # 创建质心位移张量：用于域随机化中的质心位置变化
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # 如果启用载荷质量随机化，为每个环境生成随机载荷质量
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        # 如果启用质心位移随机化，为每个环境生成随机质心位移
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        # 为每个环境创建机器人实例
        for i in range(self.num_envs):
            # 创建环境实例：在仿真中创建一个新的环境
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 获取当前环境的起始位置
            pos = self.env_origins[i].clone()
            # 在起始位置基础上添加随机偏移：在x和y方向添加±1米的随机偏移
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            # 设置机器人的初始位置
            start_pose.p = gymapi.Vec3(*pos)
                
            # 处理刚体形状属性：为当前环境设置摩擦系数、恢复系数等
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # 将处理后的刚体形状属性应用到Asset上
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 在环境中创建机器人Actor：使用Asset、位姿、名称等参数
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # 处理关节属性：为当前环境设置关节参数
            dof_props = self._process_dof_props(dof_props_asset, i)
            # 将处理后的关节属性应用到Actor上
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            # 获取Actor的刚体属性：包括质量、惯性等
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            # 只在第一个环境中保存默认刚体质量（用于后续的域随机化）
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            # 处理刚体属性：为当前环境设置质量、质心等参数
            body_props = self._process_rigid_body_props(body_props, i)
            # 将处理后的刚体属性应用到Actor上，并重新计算惯性
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # 将环境句柄添加到列表中
            self.envs.append(env_handle)
            # 将Actor句柄添加到列表中
            self.actor_handles.append(actor_handle)

        # 创建足部索引张量：用于快速访问足部刚体的状态
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        # 为每个足部刚体找到对应的索引
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # 创建被惩罚接触索引张量：用于快速访问被惩罚接触的刚体
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        # 为每个被惩罚接触的刚体找到对应的索引
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        # 创建终止接触索引张量：用于快速访问终止接触的刚体
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        # 为每个终止接触的刚体找到对应的索引
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            

    def _get_env_origins(self):
        # 获取各环境的起始位置
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        # 解析配置文件，设置各种参数
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        # 绘制调试可视化信息
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        # 初始化高度测量点，用于地形感知
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        # 初始化基座高度测量点
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        # 获取地形高度信息
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        # 获取基座相对于地形的高度
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_hip_angles(self):
        # 获取髋关节角度
        """ Computes the average hip joint angles.
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs,) containing the average hip joint angles
        """
        # 髋关节X轴索引：FL_HipX(0), FR_HipX(3), HL_HipX(6), HR_HipX(9)
        hip_joint_indices = [0, 3, 6, 9]  # 髋关节X轴（左右摆动）
        hip_angles = self.dof_pos[:, hip_joint_indices]
        # 取绝对值后求平均，因为左右腿摆动方向相反
        return torch.mean(torch.abs(hip_angles), dim=1)
    
    def _get_thigh_angles(self):
        # 获取大腿关节角度
        """ Computes the average thigh joint angles.
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs,) containing the average thigh joint angles
        """
        # 大腿关节Y轴索引：FL_HipY(1), FR_HipY(4), HL_HipY(7), HR_HipY(10)
        thigh_joint_indices = [1, 4, 7, 10]  # 大腿关节Y轴（前后摆动）
        thigh_angles = self.dof_pos[:, thigh_joint_indices]
        # 返回平均大腿关节角度
        return torch.mean(thigh_angles, dim=1)
    
    def _get_knee_angles(self):
        # 获取膝关节角度
        """ Computes the average knee joint angles.
        Returns:
            [torch.Tensor]: Tensor of shape (num_envs,) containing the average knee joint angles
        """
        # 膝关节索引：FL_Knee(2), FR_Knee(5), HL_Knee(8), HR_Knee(11)
        knee_joint_indices = [2, 5, 8, 11]  # 膝关节
        knee_angles = self.dof_pos[:, knee_joint_indices]
        # 返回平均膝关节角度
        return torch.mean(knee_angles, dim=1)

    def _get_feet_heights(self, env_ids=None):
        # 获取足部相对于地形的高度
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # 线速度跟踪奖励
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # 角速度跟踪奖励
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Z轴线速度惩罚（防止跳跃）
        # Penalize z axis base linear velocity
        reward = torch.square(self.base_lin_vel[:, 2])
        return reward
    
    def _reward_ang_vel_xy(self):
        # XY轴角速度惩罚（防止翻滚）
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # 姿态惩罚（保持直立）
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_dof_acc(self):
        # 关节加速度惩罚（平滑运动）
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        # 关节功率惩罚（节能）
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_base_height(self):
        # 基座高度惩罚（保持合适高度）
        # Penalize base height away from target
        base_height = self._get_base_heights()
        
        # 使用默认基座高度目标
        base_height_target = torch.full((self.num_envs,), self.base_height_target_normal, device=self.device)
        
        reward = torch.square(base_height - base_height_target)
        return reward
    
    def _reward_foot_clearance(self):
        """
        足部离地高度惩罚：鼓励足部在摆动时保持合适的离地高度
           - 只有在足部有横向运动时才惩罚高度误差
           - 鼓励足部在摆动时保持目标离地高度，避免拖地或过高
        """
        # 将足部位置从世界坐标系转换到相对于基座的位置
        # feet_pos: [num_envs, 4, 3] - 世界坐标系下的足部位置
        # root_states[:, 0:3]: [num_envs, 3] - 基座位置
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)  # [num_envs, 4, 3]
        
        # 将足部速度从世界坐标系转换到相对于基座的速度
        # feet_vel: [num_envs, 4, 3] - 世界坐标系下的足部速度
        # root_states[:, 7:10]: [num_envs, 3] - 基座线速度
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)  # [num_envs, 4, 3]
        
        # 初始化基座坐标系下的足部位置和速度
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        
        # 将每个足部的位置和速度从世界坐标系旋转到基座坐标系
        # 使用四元数逆旋转，将世界坐标系向量转换到基座坐标系
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        # 计算足部高度误差的平方（z方向）
        # clearance_height_target: 目标离地高度（配置文件中定义）
        # footpos_in_body_frame[:, :, 2]: 基座坐标系下足部的z坐标（高度）
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)  # [num_envs, 4]
        
        # 计算足部横向速度（xy平面速度的模长）
        # footvel_in_body_frame[:, :, :2]: 基座坐标系下足部速度的xy分量
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)  # [num_envs, 4]
        
        # 计算惩罚：高度误差² × 横向速度
        # 只有在足部有横向运动时才惩罚高度误差
        # 对4个足部的惩罚求和
        reward = torch.sum(height_error * foot_leteral_vel, dim=1)  # [num_envs]
        
        return reward
    
    def _reward_action_rate(self):
        # 动作变化率惩罚（平滑控制）
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # 运动平滑性惩罚（二阶平滑）
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # 关节力矩惩罚
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_feet_stumble(self):
        """
        脚部绊倒惩罚奖励函数
        惩罚脚部撞到垂直表面（如墙壁），检测水平方向接触力是否远大于垂直方向
        当水平接触力大于垂直接触力的4倍时，判定为撞到垂直表面
        返回值：1表示发生绊倒，0表示未发生（惩罚越大）
        """
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_dof_vel(self):
        # 关节速度惩罚
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_collision(self):
        # 碰撞惩罚
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # 回合终止惩罚
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # 关节位置限制惩罚
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # 关节速度限制惩罚
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # 力矩限制惩罚
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_joint_pos_tracking(self):
        # 关节位置跟踪奖励（保持初始姿态）
        # Encourage joints to stay close to default positions
        # 使用默认关节位置作为参考姿态
        reference_dof_pos = self.default_dof_pos.expand(self.num_envs, -1)
        joint_pos_error = torch.sum(torch.square(self.dof_pos - reference_dof_pos), dim=1)
        return joint_pos_error

    def _reward_feet_air_time(self):
        """
        足部悬空时间奖励：鼓励长步态（long steps）
        
        工作原理：
        1. 跟踪每个足部的悬空时间（从离地到接触的时间）
        2. 只在足部首次接触地面时给予奖励
        3. 奖励 = (悬空时间 - 0.5秒) * 首次接触标志
           - 悬空时间 < 0.5秒：负奖励（惩罚短步态）
           - 悬空时间 > 0.5秒：正奖励（鼓励长步态）
        4. 只在有运动命令时生效（避免静止状态下因足部悬空而获得奖励）
        
        注意：使用接触过滤（contact_filt）来处理PhysX在mesh地形上接触报告不可靠的问题
        """
        # 使用 post_physics_step 中计算的 contact_filt（类属性）
        # contact_filt 已经在 post_physics_step 中计算并存储为类属性
        
        # 检测是否是首次接触地面：
        # - feet_air_time > 0：说明之前是悬空状态
        # - contact_filt：当前帧检测到接触
        # 两者同时满足时，表示足部从悬空状态首次接触地面
        first_contact = (self.feet_air_time > 0.) * self.contact_filt  # [num_envs, 4]
        
        # 每帧增加悬空时间（所有足部都累加）
        self.feet_air_time += self.dt  # [num_envs, 4]
        
        # 计算奖励：只在首次接触地面时给予奖励
        # 奖励公式：(悬空时间 - 0.5秒) * 首次接触标志
        # - 悬空时间 < 0.5秒：负奖励（惩罚短步态，如小步快走）
        # - 悬空时间 = 0.5秒：奖励为0
        # - 悬空时间 > 0.5秒：正奖励（鼓励长步态，如大步行走或跳跃）
        # 对4个足部的奖励求和，得到每个环境的奖励
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # [num_envs]
        
        # 只在有运动命令时给予奖励
        # 使用命令速度判断（xy平面速度模长 > 0.1）
        # 这样可以避免在静止状态下因为足部悬空而获得奖励
        rew_airTime *= (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()  # [num_envs]
        
        # 重置接触足的悬空时间：如果足部接触地面，将悬空时间置为0
        # 使用 ~contact_filt 作为掩码，只有未接触的足部才保留悬空时间
        self.feet_air_time *= ~self.contact_filt  # [num_envs, 4]
        
        return rew_airTime
    
    def _reward_stand_still(self):
        # 静止惩罚（鼓励运动）
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_stand_still_pose(self):
        # 静止站立姿态奖励
        # 当速度的x,y分量小于0.15时，惩罚偏离初始姿态
        # if ||v̂_t^(x,y)|| < 0.15, -(q_t - q_t^(init))^2
        # else, 0
        vel_xy_norm = torch.norm(self.base_lin_vel[:, :2], dim=1)  # ||v̂_t^(x,y)||
        vel_threshold = 0.15  # 速度阈值
        
        # 使用默认关节位置作为参考姿态
        reference_dof_pos = self.default_dof_pos.expand(self.num_envs, -1)
        
        # 计算关节位置误差的平方和
        joint_pos_error_sq = torch.sum(torch.square(self.dof_pos - reference_dof_pos), dim=1)  # (q_t - q_t^(init))^2
        
        # 当速度小于阈值时，返回负的误差平方（惩罚偏离初始姿态）
        # 否则返回0
        reward = -joint_pos_error_sq * (vel_xy_norm < vel_threshold).float()
        
        return reward

    def _reward_feet_contact_forces(self):
        # 足部接触力惩罚
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _compute_vhip_angle(self):
        """计算VHIP摆角 θ (公式11)"""
        # 获取质心位置 (CoM)
        com_pos = self.root_states[:, 0:3]
        
        # 计算支撑中心 (CoP) - 基于足部接触力加权平均
        foot_contact_forces = self.contact_forces[:, self.feet_indices, :]
        foot_positions = self.feet_pos
        
        # 计算CoP (简化版本，只考虑垂直力)
        total_force = torch.sum(foot_contact_forces[:, :, 2], dim=1, keepdim=True)
        # 避免除零错误
        total_force = torch.clamp(total_force, min=1e-6)
        
        # 计算CoP位置 (基于接触力的加权平均)
        cop_pos = torch.sum(foot_positions * foot_contact_forces[:, :, 2:3], dim=1) / total_force
        
        # 计算摆长 (CoM到CoP的距离)
        pendulum_length = torch.norm(com_pos - cop_pos, dim=1)
        
        # 避免除零错误
        pendulum_length = torch.clamp(pendulum_length, min=1e-6)
        
        # 计算摆角 θ = arccos(||p_CoM,z|| / ||p_CoM - p_CoP||)
        com_z = com_pos[:, 2]
        cos_theta = torch.clamp(com_z / pendulum_length, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        
        return theta

    def _compute_vhip_angular_acceleration(self):
        """计算VHIP角加速度 θ̈ (公式12)"""
        # 获取当前摆角
        theta = self._compute_vhip_angle()
        
        # 获取质心位置
        com_pos = self.root_states[:, 0:3]
        
        # 计算支撑中心 (CoP)
        foot_contact_forces = self.contact_forces[:, self.feet_indices, :]
        foot_positions = self.feet_pos
        
        total_force = torch.sum(foot_contact_forces[:, :, 2], dim=1, keepdim=True)
        total_force = torch.clamp(total_force, min=1e-6)
        
        cop_pos = torch.sum(foot_positions * foot_contact_forces[:, :, 2:3], dim=1) / total_force
        
        # 计算摆长
        pendulum_length = torch.norm(com_pos - cop_pos, dim=1)
        pendulum_length = torch.clamp(pendulum_length, min=1e-6)
        
        # 重力加速度
        g = 9.81
        
        # 计算角加速度 θ̈ = -(g / ||p_CoM - p_CoP||) * sin(θ)
        theta_ddot = -(g / pendulum_length) * torch.sin(theta)
        
        return theta_ddot

    def _reward_vhip_angle(self):
        """VHIP角度奖励函数 - 基于配置参数"""
        # 计算摆角 θ
        theta = self._compute_vhip_angle()
        
        # 从配置中获取阈值
        angle_threshold = self.cfg.rewards.vhip_angle_threshold
        
        # 角度惩罚：当角度过大时给予负奖励
        angle_error = torch.abs(theta) - angle_threshold
        angle_penalty = torch.clamp(angle_error, min=0)
        
        return angle_penalty

    def _reward_vhip_angular_acceleration(self):
        """VHIP角加速度奖励函数 - 基于配置参数"""
        # 计算角加速度 θ̈
        theta_ddot = self._compute_vhip_angular_acceleration()
        
        # 从配置中获取阈值
        accel_threshold = self.cfg.rewards.vhip_angular_acc_threshold
        
        # 角加速度惩罚：当角加速度过大时给予负奖励
        accel_error = torch.abs(theta_ddot) - accel_threshold
        accel_penalty = torch.clamp(accel_error, min=0)
        
        return accel_penalty

    def _reward_hipX_limits(self):
        hip_joint_indices = [0, 3, 6, 9]  # FL_HipX, FR_HipX, HL_HipX, HR_HipX
        hip_joint_pos = self.dof_pos[:, hip_joint_indices]
        
        # 自定义的髋关节限制
        custom_limits = torch.tensor([
            [-0.17, 0.17],   # FL_HipX
            [-0.17, 0.17],   # FR_HipX  
            [-0.17, 0.17],   # HL_HipX
            [-0.17, 0.17]    # HR_HipX
        ], device=self.device)
        
        # 计算超出自定义限制的惩罚
        out_of_limits = -(hip_joint_pos - custom_limits[:, 0].unsqueeze(0)).clip(max=0.)
        out_of_limits += (hip_joint_pos - custom_limits[:, 1].unsqueeze(0)).clip(min=0.)
        
        return torch.sum(out_of_limits, dim=1)