import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import get_activation

class EnvironmentActorCritic(nn.Module):
    """
    基于DreamWaQ的环境感知Actor-Critic网络
    直接移植DreamWaQ的ActorCritic_DWAQ实现
    """
    is_recurrent = False
    
    def __init__(self, 
                 num_actor_obs,           # Actor观测维度（包含历史）
                 num_critic_obs,          # Critic观测维度
                 num_actions,             # 动作维度
                 cenet_in_dim=None,       # 环境编码器输入维度（如果为None，则使用num_actor_obs）
                 cenet_out_dim=19,        # 环境编码器输出维度 (3+16)
                 activation="elu",        # 激活函数
                 init_noise_std=1.0,      # 初始噪声标准差
                 **kwargs):
        
        if kwargs:
            print("EnvironmentActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(EnvironmentActorCritic, self).__init__()

        # 如果没有指定cenet_in_dim，使用num_actor_obs（历史观测维度）
        if cenet_in_dim is None:
            cenet_in_dim = num_actor_obs
        
        # 计算单步观测维度（假设历史长度为6）
        # 如果num_actor_obs能被6整除，则历史长度为6；否则假设为单步观测
        if num_actor_obs % 6 == 0:
            self.history_length = 6
            self.num_one_step_obs = num_actor_obs // 6
        else:
            self.history_length = 1
            self.num_one_step_obs = num_actor_obs
        
        self.cenet_in_dim = cenet_in_dim
        self.cenet_out_dim = cenet_out_dim

        self.activation = get_activation(activation)
        # 编码器和解码器使用h-swish激活函数
        self.encoder_activation = get_activation("elu")
        # Actor输入维度 = 环境编码器输出(19维) + 当前观测维度
        actor_input_dim = cenet_out_dim + self.num_one_step_obs
        critic_input_dim = num_critic_obs

        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, num_actions)
        )

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 512),
            self.activation,
            nn.Linear(512, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 1)
        )

        # 环境编码器网络 (基于DreamWaQ) - 使用h-swish激活函数
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            self.encoder_activation,
            nn.Linear(128, 64),
            self.encoder_activation,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim-3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim-3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)

        # 解码器网络 (基于DreamWaQ) - 使用h-swish激活函数
        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            self.encoder_activation,
            nn.Linear(64, 128),
            self.encoder_activation,
            nn.Linear(128, self.num_one_step_obs)  # 输出单步观测维度
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # 打印网络结构 (与原始ActorCritic保持一致)
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Environment Encoder: {self.encoder}")
        print(f"Decoder: {self.decoder}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def reparameterise(self, mean, logvar):
        """重参数化技巧"""
        # 数值稳定性：限制logvar的范围
        logvar = torch.clamp(logvar, min=-10, max=10)
        var = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(var)
        code = mean + var * code_temp
        return code
    
    def cenet_forward(self, obs_history):
        """
        环境编码器前向传播 (基于DreamWaQ)
        输入: obs_history [batch_size, cenet_in_dim] (历史观测，通常是46*6=276维)
        输出: code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent
        """
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(code)
        
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """更新动作分布"""
        mean = self.actor(observations)
        
        # NaN检查
        if torch.isnan(mean).any():
            print(f"NaN detected in Actor output: mean shape={mean.shape}, NaN count={torch.isnan(mean).sum().item()}")
            mean = torch.zeros_like(mean)
        
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, obs_history=None, **kwargs):
        """
        训练时的动作选择
        输入: observations [batch_size, num_actor_obs] (历史观测，包含当前观测)
        """
        if observations.shape[1] == self.cenet_in_dim:
            # 输入是历史观测，需要分离当前观测和历史观测
            obs_history = observations  # 历史观测
            current_obs = observations[:, :self.num_one_step_obs]  # 提取当前观测(前num_one_step_obs维)
            
            # 使用环境编码器
            code, _, decode, _, _, _, _ = self.cenet_forward(obs_history)
            # 拼接环境编码器输出和当前观测
            actor_input = torch.cat((code, current_obs), dim=-1)
        else:
            # 输入是当前观测，直接使用
            actor_input = observations
            
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """获取动作对数概率"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history=None):
        """
        推理时的动作选择
        输入: observations [batch_size, num_actor_obs] (历史观测，包含当前观测)
        """
        if observations.shape[1] == self.cenet_in_dim:
            # 输入是历史观测，需要分离当前观测和历史观测
            obs_history = observations  # 历史观测
            current_obs = observations[:, :self.num_one_step_obs]  # 提取当前观测(前num_one_step_obs维)
            
            # 使用环境编码器
            code, _, decode, _, _, _, _ = self.cenet_forward(obs_history)
            # 拼接环境编码器输出和当前观测
            actor_input = torch.cat((code, current_obs), dim=-1)
        else:
            # 输入是当前观测，直接使用
            actor_input = observations
            
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """Critic网络评估"""
        value = self.critic(critic_observations)
        return value

    def update_env_encoder(self, obs_history, speed_target=None, beta=1.0):
        """
        更新环境编码器 (基于DreamWaQ的损失函数)
        这个函数在PPO的update方法中被调用
        """
        # 前向传播
        code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent = self.cenet_forward(obs_history)
        
        # 计算损失 (基于DreamWaQ的损失函数)
        if speed_target is not None:
            vel_target = speed_target
        else:
            # 如果没有提供速度目标，使用零向量
            vel_target = torch.zeros_like(code_vel)
            
        decode_target = obs_history[:, :self.num_one_step_obs]  # 使用历史观测的第一步作为重构目标
        
        # DreamWaQ的损失函数
        vel_loss = nn.MSELoss()(code_vel, vel_target)
        recon_loss = nn.MSELoss()(decode, decode_target)
        # 修复KL损失计算：添加.mean()来平均化batch维度
        kl_loss = -0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1).mean()
        
        total_loss = vel_loss + recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss.item(),
            'vel_loss': vel_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def get_env_features(self, obs_history):
        """
        获取环境特征 (推理模式)
        输入: obs_history [batch_size, cenet_in_dim] (历史观测)
        输出: env_speed, env_latent (环境速度估计和潜在特征)
        """
        code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent = self.cenet_forward(obs_history)
        return code_vel.detach(), mean_latent.detach()

