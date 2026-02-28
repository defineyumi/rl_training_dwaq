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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import EnvironmentActorCritic
from rsl_rl.storage import RolloutStorage

class EnvironmentPPO:
    """
    环境编码器PPO算法
    在标准PPO基础上，增加环境编码器的训练
    """
    actor_critic: EnvironmentActorCritic
    
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 env_encoder_lr=1e-3,
                 env_encoder_beta=1.0,
                 # 基于性能的动态熵系数参数
                 use_dynamic_entropy=False,
                 lin_vel_threshold=0.7,
                 ang_vel_threshold=0.3,
                 terrain_level_threshold=5,
                 high_entropy_coef=0.02,
                 low_entropy_coef=0.01,
                 softmax_temperature=2.0,
                 ):

        self.device = device

        self.desired_kl = desired_kl  # 期望KL散度
        self.schedule = schedule      # 学习率调度
        self.learning_rate = learning_rate  # 学习率

        # PPO组件
        self.actor_critic = actor_critic  # Actor-Critic网络
        self.actor_critic.to(self.device)
        self.storage = None  # 经验存储
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO参数
        self.clip_param = clip_param  # 裁剪参数
        self.num_learning_epochs = num_learning_epochs  # 学习轮数
        self.num_mini_batches = num_mini_batches  # 小批量数
        self.value_loss_coef = value_loss_coef  # 价值损失系数
        self.entropy_coef = entropy_coef  # 熵系数
        self.gamma = gamma  # 折扣因子
        self.lam = lam  # GAE参数
        self.max_grad_norm = max_grad_norm  # 梯度裁剪
        self.use_clipped_value_loss = use_clipped_value_loss  # 裁剪价值损失

        # 环境编码器参数
        self.env_encoder_lr = env_encoder_lr  # 环境编码器学习率
        self.env_encoder_beta = env_encoder_beta  # KL散度权重
        
        # 基于性能的动态熵系数参数
        self.use_dynamic_entropy = use_dynamic_entropy  # 是否启用动态熵系数
        self.lin_vel_threshold = lin_vel_threshold  # 线速度跟踪阈值
        self.ang_vel_threshold = ang_vel_threshold  # 角速度跟踪阈值
        self.terrain_level_threshold = terrain_level_threshold  # 地形等级阈值
        self.high_entropy_coef = high_entropy_coef  # 高熵系数
        self.low_entropy_coef = low_entropy_coef  # 低熵系数
        self.softmax_temperature = softmax_temperature  # Softmax温度参数
        
        # 当前熵系数
        self.current_entropy_coef = entropy_coef

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        """
        初始化经验回放缓冲区
        
        Args:
            num_envs: 环境数量
            num_transitions_per_env: 每个环境收集的转换数量
            actor_obs_shape: Actor网络观测空间形状
            critic_obs_shape: Critic网络观测空间形状
            action_shape: 动作空间形状
        """
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        """设置网络为测试模式（关闭dropout等）"""
        self.actor_critic.test()
    
    def update_dynamic_entropy_coef(self, performance_metrics=None):
        """
        基于性能指标更新动态熵系数（递减式）
        
        Args:
            performance_metrics: 性能指标字典，包含：
                - lin_vel_tracking: 线速度跟踪性能
                - ang_vel_tracking: 角速度跟踪性能  
                - terrain_level: 当前地形等级
        """
        if not self.use_dynamic_entropy or performance_metrics is None:
            return self.current_entropy_coef
        
        # 获取性能指标
        lin_vel_tracking = performance_metrics.get('lin_vel_tracking', 0.0)
        ang_vel_tracking = performance_metrics.get('ang_vel_tracking', 0.0)
        terrain_level = performance_metrics.get('terrain_level', 0)
        
        # 计算各项指标距离目标的差距
        lin_vel_gap = max(0, self.lin_vel_threshold - lin_vel_tracking)  # 线速度差距
        ang_vel_gap = max(0, self.ang_vel_threshold - ang_vel_tracking)   # 角速度差距
        terrain_gap = max(0, self.terrain_level_threshold - terrain_level) # 地形等级差距
        
        # 归一化差距 (0-1之间)
        norm_lin_gap = lin_vel_gap / self.lin_vel_threshold if self.lin_vel_threshold > 0 else 0
        norm_ang_gap = ang_vel_gap / self.ang_vel_threshold if self.ang_vel_threshold > 0 else 0
        norm_terrain_gap = terrain_gap / self.terrain_level_threshold if self.terrain_level_threshold > 0 else 0
        
        # 将差距转换为tensor
        gaps = torch.tensor([norm_lin_gap, norm_ang_gap, norm_terrain_gap], dtype=torch.float32)
        
        # 使用softmax计算归一化权重（差距越大，权重越高）
        weights = F.softmax(gaps / self.softmax_temperature, dim=0)
        
        # 计算加权平均差距
        weighted_gap = torch.sum(weights * gaps).item()
        
        # 基于加权差距计算熵系数（差距越大，熵系数越高）
        # 当weighted_gap=1时（所有指标都未达标），使用高熵系数
        # 当weighted_gap=0时（所有指标都达标），使用低熵系数
        self.current_entropy_coef = self.low_entropy_coef + weighted_gap * (self.high_entropy_coef - self.low_entropy_coef)
        
        return self.current_entropy_coef
    
    def train_mode(self):
        """设置网络为训练模式（启用dropout等）"""
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        """
        根据观测选择动作
        
        Args:
            obs: Actor网络观测
            critic_obs: Critic网络观测
            
        Returns:
            actions: 选择的动作
        """
        # 如果是循环网络，获取隐藏状态
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # 计算动作和价值函数
        self.transition.actions = self.actor_critic.act(obs).detach()  # 从策略网络获取动作
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()  # 从价值网络获取价值估计
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()  # 动作对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach()  # 动作均值
        self.transition.action_sigma = self.actor_critic.action_std.detach()  # 动作标准差
        
        # 记录观测数据
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        """
        处理环境步骤，存储转换数据
        
        Args:
            rewards: 奖励
            dones: 回合结束标志
            infos: 额外信息
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # 处理超时情况（Bootstrap）
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # 存储转换数据
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_values):
        """
        计算回报和优势函数
        
        Args:
            last_values: 最后一步的价值估计
        """
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """
        执行PPO更新，包括环境编码器训练
        
        执行以下步骤：
        1. 从存储缓冲区获取小批量数据
        2. 计算KL散度并调整学习率（如果启用自适应学习率）
        3. 计算代理损失（PPO的核心）
        4. 计算价值函数损失
        5. 更新环境编码器
        6. 执行梯度下降更新
        
        Returns:
            mean_value_loss: 平均价值函数损失
            mean_surrogate_loss: 平均代理损失
            mean_env_loss: 平均环境编码器损失
        """
        # 初始化损失累积器
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_env_loss = 0
        
        # 根据网络类型选择数据生成器
        if self.actor_critic.is_recurrent:
            # 循环网络需要处理序列数据
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            # 前馈网络使用标准小批量生成器
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        # 遍历所有小批量数据进行训练
        for obs_batch, critic_obs_batch, obs_hist_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            # 前向传播：计算当前策略的动作和价值
            self.actor_critic.act(obs_batch, masks=None, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)  # 当前策略的动作对数概率
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=None, hidden_states=hid_states_batch[1])  # 当前价值估计
            mu_batch = self.actor_critic.action_mean  # 当前动作均值
            sigma_batch = self.actor_critic.action_std  # 当前动作标准差
            entropy_batch = self.actor_critic.entropy  # 当前策略熵

            # KL散度计算和自适应学习率调整
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():  # 不计算梯度，仅用于学习率调整
                    # 计算KL散度：衡量新旧策略之间的差异
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                        (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    # 根据KL散度调整学习率
                    if kl_mean > self.desired_kl * 2.0:
                        # KL散度过大，策略变化太大，降低学习率
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        # KL散度过小，策略变化太小，提高学习率
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    # 更新优化器的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # PPO代理损失计算（核心部分）
            # 计算重要性采样比率：π_new(a|s) / π_old(a|s)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            
            # 未裁剪的代理损失
            surrogate = -torch.squeeze(advantages_batch) * ratio
            
            # 裁剪的代理损失：限制比率在[1-clip_param, 1+clip_param]范围内
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            
            # 取两者中的较大值（保守更新）
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 价值函数损失计算
            if self.use_clipped_value_loss:
                # 使用裁剪的价值函数损失（类似PPO的裁剪机制）
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)  # 未裁剪的损失
                value_losses_clipped = (value_clipped - returns_batch).pow(2)  # 裁剪的损失
                value_loss = torch.max(value_losses, value_losses_clipped).mean()  # 取较大值
            else:
                # 标准MSE损失
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 环境编码器VAE损失计算 (基于DreamWaQ)
            autoenc_loss = 0.0
            if hasattr(self.actor_critic, 'cenet_forward'):
                # 使用当前特权观测作为速度目标，而不是前一步的
                # 检查是否是历史观测（维度等于cenet_in_dim或num_actor_obs）
                obs_hist_batch = None
                if hasattr(self.actor_critic, 'cenet_in_dim') and obs_batch.shape[1] == self.actor_critic.cenet_in_dim:
                    obs_hist_batch = obs_batch
                elif hasattr(self.actor_critic, 'num_actor_obs') and obs_batch.shape[1] == self.actor_critic.num_actor_obs:
                    obs_hist_batch = obs_batch
                
                if obs_hist_batch is not None:
                    # 使用DreamWaQ的VAE损失计算方式
                    code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent = self.actor_critic.cenet_forward(obs_hist_batch)
                    
                    # 从当前特权观测中提取速度目标 (简化版本)
                    # 特权观测结构：单步观测(46维) + 基座线速度(3维) + 外力(3维) + 扫描点(187维)
                    num_one_step_obs = self.actor_critic.num_one_step_obs if hasattr(self.actor_critic, 'num_one_step_obs') else 46
                    vel_target = critic_obs_batch[:, num_one_step_obs:num_one_step_obs+3]  # 基座线速度
                    # 解码器目标是单步观测，从历史观测中提取第一步
                    decode_target = obs_hist_batch[:, :num_one_step_obs]  # 历史观测的第一步
                    
                    vel_target.requires_grad = False
                    decode_target.requires_grad = False
                    
                    # 计算VAE损失 (基于DreamWaQ)
                    vel_loss = nn.MSELoss()(code_vel, vel_target)
                    recon_loss = nn.MSELoss()(decode, decode_target)
                    
                    # 数值稳定的KL损失计算
                    kl_loss = -0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - torch.exp(torch.clamp(logvar_latent, min=-10, max=10)))
                    
                    # 检查重构损失
                    if torch.isnan(recon_loss):
                        print(f"Recon loss NaN: decode range=[{decode.min().item():.4f}, {decode.max().item():.4f}], target range=[{decode_target.min().item():.4f}, {decode_target.max().item():.4f}]")
                        recon_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    
                    # NaN检查
                    if torch.isnan(vel_loss) or torch.isnan(recon_loss) or torch.isnan(kl_loss):
                        print(f"NaN detected in VAE loss: vel_loss={vel_loss.item()}, recon_loss={recon_loss.item()}, kl_loss={kl_loss.item()}")
                        autoenc_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:
                        # 原始环境损失（无裁剪）
                        autoenc_loss = (vel_loss + recon_loss + self.env_encoder_beta * kl_loss) / self.num_mini_batches

            # 总损失：代理损失 + 价值函数损失 - 熵损失 + VAE损失
            # 使用动态熵系数
            current_entropy_coef = self.current_entropy_coef if self.use_dynamic_entropy else self.entropy_coef
            loss = surrogate_loss + self.value_loss_coef * value_loss - current_entropy_coef * entropy_batch.mean() + autoenc_loss

            # 梯度下降步骤
            self.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)  # 梯度裁剪防止梯度爆炸
            self.optimizer.step()  # 更新参数

            # 累积损失用于统计
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_env_loss += autoenc_loss

        # 计算平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_env_loss /= num_updates
        
        # 清空存储缓冲区，准备下一轮数据收集
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_env_loss
