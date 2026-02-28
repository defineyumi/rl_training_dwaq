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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096  # 并行环境数量
        num_one_step_observations = 45  # 单步观测维度（3维运动指令+3维基座角速度+3维重力+12维关节位置+12维关节速度+12维动作）
        num_observations = num_one_step_observations * 6  # 总观测维度（包含历史观测，6步历史）
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # 单步特权观测维度（额外包含：基座线速度、外力、扫描点）
        num_privileged_obs = num_one_step_privileged_obs * 1  # 总特权观测维度（用于非对称训练，Critic使用特权信息）
        num_actions = 12  # 动作维度（12个关节）
        env_spacing = 3.  # 环境间距 [m]（使用高度场/三角网格时不使用）
        send_timeouts = True  # 是否发送超时信息给算法
        episode_length_s = 20  # 每个回合的长度 [秒]

    class terrain:
        mesh_type = 'trimesh'  # 地形网格类型：none（无地形）、plane（平面）、heightfield（高度场）、trimesh（三角网格）
        horizontal_scale = 0.1  # 水平缩放比例 [m]（地形分辨率）
        vertical_scale = 0.005  # 垂直缩放比例 [m]（地形高度变化幅度）
        border_size = 25  # 边界大小 [m]（地形边界区域）
        curriculum = True  # 是否启用地形课程学习（逐步增加难度）
        static_friction = 1.0  # 静摩擦系数
        dynamic_friction = 1.0  # 动摩擦系数
        restitution = 0.  # 恢复系数（弹性碰撞系数，0为完全非弹性）
        # 粗糙地形专用参数：
        measure_heights = True  # 是否测量高度（用于观测）
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # X方向测量点位置 [m]（1m×1.6m矩形，不含中心线）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # Y方向测量点位置 [m]
        selected = False  # 是否选择单一地形类型（如果为True，将使用terrain_kwargs中的参数）
        terrain_kwargs = None  # 选定地形的参数字典
        max_init_terrain_level = 0  # 初始地形等级（课程学习的起始状态）
        terrain_length = 8.  # 地形长度 [m]
        terrain_width = 8.  # 地形宽度 [m]
        num_rows = 10  # 地形行数（难度等级数）
        num_cols = 20  # 地形列数（地形类型数）
        # 地形类型比例：[平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散障碍]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # 仅用于三角网格：
        slope_treshold = 0.75  # 坡度阈值（超过此阈值的斜坡将被修正为垂直表面）

    class commands:
        curriculum = True  # 是否启用速度指令课程学习（逐步增加速度范围）
        max_curriculum = 2.0  # 速度课程学习的最大速度 [m/s]
        num_commands = 4  # 指令数量（线速度x、线速度y、角速度yaw、航向角）
        resampling_time = 10.  # 指令重新采样时间 [s]（在此时间后更换指令）
        heading_command = True  # 是否使用航向指令模式（如果为True，从航向误差计算角速度指令）
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # X方向线速度范围 [m/s]（最小值，最大值）
            lin_vel_y = [-1.0, 1.0]  # Y方向线速度范围 [m/s]（最小值，最大值）
            ang_vel_yaw = [-3.14, 3.14]  # 偏航角速度范围 [rad/s]（最小值，最大值）
            heading = [-3.14, 3.14]  # 航向角范围 [rad]（最小值，最大值）

    class init_state:
        pos = [0.0, 0.0, 1.]  # 初始位置 [m]（x, y, z）
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始旋转 [四元数]（x, y, z, w）
        lin_vel = [0.0, 0.0, 0.0]  # 初始线速度 [m/s]（x, y, z）
        ang_vel = [0.0, 0.0, 0.0]  # 初始角速度 [rad/s]（x, y, z）
        default_joint_angles = {  # 默认关节角度（当动作=0.0时的目标角度）
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P'  # 控制类型：P（位置控制）、V（速度控制）、T（力矩控制）
        # PD控制器参数：
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # 关节刚度 [N*m/rad]（比例增益KP）
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # 关节阻尼 [N*m*s/rad]（微分增益KD）
        # 动作缩放：目标角度 = action_scale * action + default_angle
        action_scale = 0.5  # 动作缩放系数
        # 降采样：每个策略时间步内的控制动作更新次数（策略频率 = 仿真频率 / decimation）
        decimation = 4  # 降采样倍数
        hip_reduction = 1.0  # 髋关节缩放系数

    class asset:
        file = ""  # URDF/MJCF文件路径
        name = "legged_robot"  # 机器人actor名称
        foot_name = "None"  # 足部刚体名称（用于索引刚体状态和接触力张量）
        penalize_contacts_on = []  # 需要惩罚接触的刚体列表
        terminate_after_contacts_on = []  # 接触后终止回合的刚体列表
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = True  # 是否合并固定关节连接的刚体（可通过添加dont_collapse="true"保留特定固定关节）
        fix_base_link = False  # 是否固定机器人基座
        default_dof_drive_mode = 3  # 默认关节驱动模式（0=无驱动，1=位置目标，2=速度目标，3=力矩）
        self_collisions = 0  # 自碰撞设置（1=禁用，0=启用，位掩码过滤器）
        replace_cylinder_with_capsule = True  # 是否用胶囊体替换圆柱体碰撞（提高仿真速度和稳定性）
        flip_visual_attachments = True  # 是否翻转视觉附件（某些.obj网格需要从y-up翻转到z-up）
        
        density = 0.001  # 材料密度 [kg/m³]
        angular_damping = 0.  # 角速度阻尼系数
        linear_damping = 0.  # 线速度阻尼系数
        max_angular_velocity = 1000.  # 最大角速度 [rad/s]
        max_linear_velocity = 1000.  # 最大线速度 [m/s]
        armature = 0.  # 关节惯性（电机惯性）
        thickness = 0.01  # 碰撞体厚度 [m]

    class domain_rand:
        randomize_payload_mass = True  # 是否随机化负载质量
        payload_mass_range = [-1, 2]  # 负载质量范围 [kg]（最小值，最大值）

        randomize_com_displacement = True  # 是否随机化质心偏移
        com_displacement_range = [-0.05, 0.05]  # 质心偏移范围 [m]（最小值，最大值）

        randomize_link_mass = False  # 是否随机化连杆质量
        link_mass_range = [0.9, 1.1]  # 连杆质量范围（相对于原始质量的倍数）
        
        randomize_friction = True  # 是否随机化摩擦系数
        friction_range = [0.2, 1.25]  # 摩擦系数范围（最小值，最大值）
        
        randomize_restitution = False  # 是否随机化恢复系数
        restitution_range = [0., 1.0]  # 恢复系数范围（最小值，最大值）
        
        randomize_motor_strength = True  # 是否随机化电机强度
        motor_strength_range = [0.9, 1.1]  # 电机强度范围（相对于原始强度的倍数）
        
        randomize_kp = True  # 是否随机化比例增益KP
        kp_range = [0.9, 1.1]  # KP范围（相对于原始KP的倍数）
        
        randomize_kd = True  # 是否随机化微分增益KD
        kd_range = [0.9, 1.1]  # KD范围（相对于原始KD的倍数）
        
        randomize_initial_joint_pos = True  # 是否随机化初始关节位置
        initial_joint_pos_range = [0.5, 1.5]  # 初始关节位置范围（相对于默认角度的倍数）
        
        disturbance = True  # 是否启用外部扰动
        disturbance_range = [-30.0, 30.0]  # 扰动力范围 [N]（最小值，最大值）
        disturbance_interval = 8  # 扰动间隔 [s]
        
        push_robots = True  # 是否启用推动机器人
        push_interval_s = 16  # 推动间隔 [s]
        max_push_vel_xy = 1.  # 最大推动速度 [m/s]（XY平面）

        delay = True  # 是否启用动作延迟

    class rewards:
        class scales:
            termination = -0.0  # 终止惩罚权重（回合提前结束的惩罚）
            tracking_lin_vel = 1.0  # 线速度跟踪奖励权重
            tracking_ang_vel = 0.5  # 角速度跟踪奖励权重
            lin_vel_z = -2.0  # Z方向线速度惩罚权重（防止垂直运动）
            ang_vel_xy = -0.05  # XY方向角速度惩罚权重（防止翻滚）
            orientation = -0.  # 姿态惩罚权重（保持水平姿态）
            torques = -0.00001  # 力矩惩罚权重（减少能耗）
            dof_vel = -0.  # 关节速度惩罚权重
            dof_acc = -2.5e-7  # 关节加速度惩罚权重（平滑性）
            base_height = -0.  # 基座高度惩罚权重（保持目标高度）
            feet_air_time = 1.0  # 足部腾空时间奖励权重（鼓励步态）
            collision = -1.  # 碰撞惩罚权重（避免与障碍物碰撞）
            feet_stumble = -0.0  # 足部绊倒惩罚权重
            action_rate = -0.01  # 动作变化率惩罚权重（平滑性）
            stand_still = -0.  # 静止惩罚权重（鼓励运动）

        only_positive_rewards = True  # 是否只保留正奖励（如果为True，负奖励会被裁剪到0，避免过早终止问题）
        tracking_sigma = 0.25  # 跟踪奖励的高斯参数：tracking_reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # 关节位置软限制：URDF限制的百分比，超过此值会被惩罚
        soft_dof_vel_limit = 1.  # 关节速度软限制：URDF限制的百分比，超过此值会被惩罚
        soft_torque_limit = 1.  # 力矩软限制：URDF限制的百分比，超过此值会被惩罚
        base_height_target = 1.  # 基座目标高度 [m]
        max_contact_force = 100.  # 最大接触力 [N]，超过此值会被惩罚
        clearance_height_target = 0.09  # 足部离地目标高度 [m]（用于足部清除奖励）

    class normalization:
        class obs_scales:
            lin_vel = 2.0  # 线速度观测缩放系数
            ang_vel = 0.25  # 角速度观测缩放系数
            dof_pos = 1.0  # 关节位置观测缩放系数
            dof_vel = 0.05  # 关节速度观测缩放系数
            height_measurements = 5.0  # 高度测量观测缩放系数
        clip_observations = 100.  # 观测值裁剪范围（-clip_observations到+clip_observations）
        clip_actions = 100.  # 动作值裁剪范围（-clip_actions到+clip_actions）

    class noise:
        add_noise = True  # 是否添加观测噪声（域随机化）
        noise_level = 1.0  # 噪声水平（缩放其他噪声值）
        class noise_scales:
            dof_pos = 0.01  # 关节位置噪声标准差
            dof_vel = 1.5  # 关节速度噪声标准差
            lin_vel = 0.1  # 线速度噪声标准差
            ang_vel = 0.2  # 角速度噪声标准差
            gravity = 0.05  # 重力噪声标准差
            height_measurements = 0.1  # 高度测量噪声标准差

    # 查看器相机设置：
    class viewer:
        ref_env = 0  # 参考环境索引（用于相机跟踪）
        pos = [10, 0, 6]  # 相机位置 [m]（x, y, z）
        lookat = [11., 5, 3.]  # 相机朝向目标点 [m]（x, y, z）

    class sim:
        dt = 0.005  # 仿真时间步长 [s]
        substeps = 1  # 子步数（每个时间步内的物理子步数）
        gravity = [0., 0., -9.81]  # 重力加速度 [m/s²]（x, y, z）
        up_axis = 1  # 上方向轴（0=y轴，1=z轴）

        class physx:
            num_threads = 10  # PhysX物理引擎线程数
            solver_type = 1  # 求解器类型（0=PGS投影高斯-赛德尔，1=TGS时间步高斯-赛德尔）
            num_position_iterations = 4  # 位置迭代次数（约束求解迭代）
            num_velocity_iterations = 0  # 速度迭代次数（约束求解迭代）
            contact_offset = 0.01  # 接触偏移 [m]（接触检测距离）
            rest_offset = 0.0  # 静止偏移 [m]（物体静止时的接触距离）
            bounce_threshold_velocity = 0.5  # 弹跳阈值速度 [m/s]（超过此速度才考虑弹跳）
            max_depenetration_velocity = 1.0  # 最大去穿透速度 [m/s]（解决穿透问题的最大速度）
            max_gpu_contact_pairs = 2**23  # 最大GPU接触对数量（2^24用于8000+环境）
            default_buffer_size_multiplier = 5  # 默认缓冲区大小倍数
            contact_collection = 2  # 接触收集模式（0=从不，1=最后子步，2=所有子步，默认=2）

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1  # 随机种子
    runner_class_name = 'OnPolicyRunner'  # 训练运行器类名
    class policy:
        init_noise_std = 1.0  # 初始动作噪声标准差
        actor_hidden_dims = [512, 256, 128]  # Actor网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]  # Critic网络隐藏层维度
        activation = 'elu'  # 激活函数类型（可选：elu, relu, selu, crelu, lrelu, tanh, sigmoid）
        # 仅用于'ActorCriticRecurrent'（循环网络）：
        # rnn_type = 'lstm'  # RNN类型（LSTM）
        # rnn_hidden_size = 512  # RNN隐藏层大小
        # rnn_num_layers = 1  # RNN层数
        
    class algorithm:
        # 训练参数
        value_loss_coef = 1.0  # 价值函数损失系数（在总损失中的权重）
        use_clipped_value_loss = True  # 是否使用裁剪的价值函数损失（类似PPO裁剪机制）
        clip_param = 0.2  # PPO裁剪参数（限制策略更新幅度）
        entropy_coef = 0.01  # 熵损失系数（鼓励探索，值越大探索越多）
        num_learning_epochs = 5  # 每次更新的学习轮数（对同一批数据重复训练的次数）
        num_mini_batches = 4  # 小批量数量（小批量大小 = num_envs * num_steps_per_env / num_mini_batches）
        learning_rate = 1.e-3  # 学习率（优化器学习率）
        schedule = 'adaptive'  # 学习率调度策略（'adaptive'=自适应，'fixed'=固定）
        gamma = 0.99  # 折扣因子（未来奖励的折扣率）
        lam = 0.95  # GAE参数（广义优势估计的λ参数）
        desired_kl = 0.01  # 目标KL散度（用于自适应学习率调整）
        max_grad_norm = 1.  # 最大梯度范数（梯度裁剪阈值，防止梯度爆炸）

    class runner:
        policy_class_name = 'ActorCritic'  # 策略网络类名
        algorithm_class_name = 'PPO'  # 算法类名
        num_steps_per_env = 100  # 每个环境每次迭代的步数
        max_iterations = 200000  # 最大迭代次数（策略更新次数）

        # 日志记录
        save_interval = 20  # 保存间隔（每N次迭代检查并保存模型）
        experiment_name = 'test'  # 实验名称
        run_name = ''  # 运行名称（如果为空则自动生成）
        # 加载和恢复训练
        resume = False  # 是否恢复训练
        load_run = -1  # 要加载的运行编号（-1=最后一次运行）
        checkpoint = -1  # 要加载的检查点（-1=最后保存的模型）
        resume_path = None  # 恢复路径（从load_run和checkpoint自动更新）