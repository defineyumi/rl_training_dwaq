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

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    """
    PPO (Proximal Policy Optimization) 算法实现
    
    PPO是一种策略梯度强化学习算法，通过限制策略更新的幅度来保证训练的稳定性。
    主要特点：
    1. 使用重要性采样和裁剪机制
    2. 支持自适应学习率调整
    3. 结合价值函数和策略网络
    """
    actor_critic: ActorCritic
    
    def __init__(self,
                 actor_critic,                    # Actor-Critic网络
                 num_learning_epochs=1,           # 每次更新的学习轮数
                 num_mini_batches=1,              # 小批量数量
                 clip_param=0.2,                  # PPO裁剪参数
                 gamma=0.998,                     # 折扣因子
                 lam=0.95,                        # GAE参数
                 value_loss_coef=1.0,             # 价值函数损失系数
                 entropy_coef=0.0,                # 熵损失系数
                 learning_rate=1e-3,              # 学习率
                 max_grad_norm=1.0,               # 梯度裁剪阈值
                 use_clipped_value_loss=True,     # 是否使用裁剪的价值函数损失
                 schedule="fixed",                # 学习率调度策略
                 desired_kl=0.01,                 # 目标KL散度
                 device='cpu',                    # 计算设备
                 # 动态熵机制参数
                 use_dynamic_entropy=False,       # 是否启用动态熵系数
                 lin_vel_threshold=0.7,           # 线速度跟踪阈值
                 ang_vel_threshold=0.3,           # 角速度跟踪阈值
                 terrain_level_threshold=5,       # 地形等级阈值
                 high_entropy_coef=0.02,          # 高熵系数
                 low_entropy_coef=0.01,           # 低熵系数
                 softmax_temperature=2.0,         # Softmax温度参数
                 ):
        """
        初始化PPO算法
        
        Args:
            actor_critic: Actor-Critic网络，包含策略网络和价值网络
            num_learning_epochs: 每次更新时对数据进行的学习轮数
            num_mini_batches: 将数据分成的小批量数量
            clip_param: PPO裁剪参数，限制策略更新幅度
            gamma: 折扣因子，用于计算未来奖励的现值
            lam: GAE参数，用于计算优势函数
            value_loss_coef: 价值函数损失在总损失中的权重
            entropy_coef: 熵损失在总损失中的权重（鼓励探索）
            learning_rate: 优化器学习率
            max_grad_norm: 梯度裁剪的最大范数
            use_clipped_value_loss: 是否对价值函数损失进行裁剪
            schedule: 学习率调度策略（'fixed'或'adaptive'）
            desired_kl: 目标KL散度，用于自适应学习率调整
            device: 计算设备（'cpu'或'cuda'）
        """
        # 设置计算设备
        self.device = device

        # 学习率调整相关参数
        self.desired_kl = desired_kl              # 目标KL散度，用于判断策略更新幅度
        self.schedule = schedule                  # 学习率调度策略
        self.learning_rate = learning_rate        # 当前学习率

        # PPO核心组件
        self.actor_critic = actor_critic          # Actor-Critic网络
        self.actor_critic.to(self.device)         # 将网络移动到指定设备
        self.storage = None                       # 经验回放缓冲区，稍后初始化
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)  # Adam优化器
        self.transition = RolloutStorage.Transition()  # 单步转换数据

        # PPO算法参数
        self.clip_param = clip_param              # PPO裁剪参数，限制策略更新幅度
        self.num_learning_epochs = num_learning_epochs  # 每次更新的学习轮数
        self.num_mini_batches = num_mini_batches  # 小批量数量
        self.value_loss_coef = value_loss_coef    # 价值函数损失系数
        self.entropy_coef = entropy_coef          # 熵损失系数（鼓励探索）
        self.gamma = gamma                        # 折扣因子
        self.lam = lam                            # GAE参数
        self.max_grad_norm = max_grad_norm        # 梯度裁剪阈值
        self.use_clipped_value_loss = use_clipped_value_loss  # 是否使用裁剪的价值函数损失
        
        # 动态熵机制参数
        self.use_dynamic_entropy = use_dynamic_entropy  # 是否启用动态熵系数
        self.lin_vel_threshold = lin_vel_threshold      # 线速度跟踪阈值
        self.ang_vel_threshold = ang_vel_threshold       # 角速度跟踪阈值
        self.terrain_level_threshold = terrain_level_threshold  # 地形等级阈值
        self.high_entropy_coef = high_entropy_coef      # 高熵系数
        self.low_entropy_coef = low_entropy_coef         # 低熵系数
        self.softmax_temperature = softmax_temperature  # Softmax温度参数
        
        # 当前熵系数（动态调整）
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
    
    def train_mode(self):
        """设置网络为训练模式（启用dropout等）"""
        self.actor_critic.train()
    
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
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()  # 动作的对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach()  # 动作均值
        self.transition.action_sigma = self.actor_critic.action_std.detach()  # 动作标准差
        
        # 记录观测数据（在环境步骤之前）
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        """
        处理环境步骤，记录转换数据
        
        Args:
            rewards: 奖励
            dones: 回合结束标志
            infos: 额外信息
        """
        # 记录奖励和结束标志
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # 处理超时情况的自举（Bootstrap）
        # 如果回合因超时结束，使用当前价值估计作为奖励的一部分
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # 将转换数据添加到存储缓冲区
        self.storage.add_transitions(self.transition)
        # 清空当前转换数据
        self.transition.clear()
        # 重置Actor-Critic网络状态（用于循环网络）
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_values):
        """
        计算回报和优势函数
        
        Args:
            last_values: 最后一步的价值估计
        """
        # 使用GAE计算回报和优势函数
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """
        PPO算法的主要更新函数
        
        执行以下步骤：
        1. 从存储缓冲区获取小批量数据
        2. 计算KL散度并调整学习率（如果启用自适应学习率）
        3. 计算代理损失（PPO的核心）
        4. 计算价值函数损失
        5. 执行梯度下降更新
        
        Returns:
            mean_value_loss: 平均价值函数损失
            mean_surrogate_loss: 平均代理损失
        """
        # 初始化损失累积器
        mean_value_loss = 0
        mean_surrogate_loss = 0
        
        # 根据网络类型选择数据生成器
        if self.actor_critic.is_recurrent:
            # 循环网络需要处理序列数据
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            # 前馈网络使用标准小批量生成器
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        # 遍历所有小批量数据进行训练
        for obs_batch, critic_obs_batch, prev_critic_obs_batch, obs_hist_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            # 前向传播：计算当前策略的动作和价值
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)  # 当前策略的动作对数概率
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])  # 当前价值估计
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

            # 总损失：代理损失 + 价值函数损失 - 熵损失（熵损失鼓励探索）
            # 使用动态熵系数或固定熵系数
            current_entropy_coef = self.current_entropy_coef if self.use_dynamic_entropy else self.entropy_coef
            loss = surrogate_loss + self.value_loss_coef * value_loss - current_entropy_coef * entropy_batch.mean()

            # 梯度下降步骤
            self.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播计算梯度
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)  # 梯度裁剪防止梯度爆炸
            self.optimizer.step()  # 更新参数

            # 累积损失用于统计
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        # 计算平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        
        # 清空存储缓冲区，准备下一轮数据收集
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
