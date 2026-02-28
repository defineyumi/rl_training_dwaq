from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X30RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096  # 环境数量
        num_one_step_observations = 45  # 单步观测维度
        use_history_obs = False # 是否使用历史信息
        # 根据开关动态设置观测维度
        if use_history_obs:
            num_observations = num_one_step_observations * 6  # 270维
        else:
            num_observations = num_one_step_observations      # 45维
        # num_observations = num_one_step_observations  # 总观测维度（只使用单步观测）
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # 单步特权观测维度（额外包含：基座线速度、外部力、扫描点）
        num_privileged_obs = num_one_step_privileged_obs  # 总特权观测维度（只使用单步特权观测）
        num_actions = 12  # 动作维度
        env_spacing = 3.  # 环境间距（不用于高度场/三角网格）
        send_timeouts = True  # 向算法发送超时信息
        episode_length_s = 20  # 回合长度（秒）

        class commands( LeggedRobotCfg.commands ):
            curriculum = True
            max_curriculum = 2 #最大难度 - 限制最大速度为2m/s
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            class ranges:
                lin_vel_x = [-1.0, 1.0] # min max [m/s] - 速度范围，主要用于站立测试
                lin_vel_y = [-1.0, 1.0]   # min max [m/s] - 横向速度范围
                ang_vel_yaw = [-1.0 , 1.0]   # min max [rad/s] - 角速度范围
                heading = [-1.0, 1.0] #这两个值=3.14的时候转弯超级烂，可能是要小一点比较好


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m] - 调整为与base_height_target匹配
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.0,   # [rad] 顺时针为正
            'HL_HipX_joint': 0.0,   # [rad]
            'FR_HipX_joint': 0.0,   # [rad]
            'HR_HipX_joint': 0.0,   # [rad]

            'FL_HipY_joint': -0.6,   # [rad] 逆时针为正
            'HL_HipY_joint': -0.6,   # [rad] 
            'FR_HipY_joint': -0.6,   # [rad] 
            'HR_HipY_joint': -0.6,   # [rad] 

            'FL_Knee_joint': 1.3,   # [rad] 顺时正为正
            'HL_Knee_joint': 1.3,   # [rad] 
            'FR_Knee_joint': 1.3,   # [rad] 
            'HR_Knee_joint': 1.3,   # [rad] 
        }

    class control( LeggedRobotCfg.control ):
        # PD控制器参数:
        control_type = 'P'  # 控制类型：P表示比例控制（只学习应该到达的角度，不学习到达这个角度需要多大的力）
        stiffness = {'joint': 120.0}  # 关节刚度 [N*m/rad] - 匹配部署参数
        damping = {'joint': 1.6}     # 关节阻尼 [N*m*s/rad] - 匹配部署参数
        # 动作缩放：目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25  # 动作缩放系数
        # 抽取：每个策略时间步长内仿真时间步长的控制动作更新次数
        decimation = 4  # 控制频率抽取因子
        hip_reduction = 1.0  # 髋关节减速比

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X30/urdf/X30.urdf'
        name = "X30"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]  # 接触时会被惩罚的部件（大腿、小腿）
        terminate_after_contacts_on = ["TORSO"]  # 接触时会终止回合的部件（躯干）
        privileged_contacts_on = ["TORSO", "THIGH", "SHANK"]  # 特权观测中包含接触信息的部件
        self_collisions = 1  # 自碰撞检测：1表示禁用，0表示启用（位掩码过滤器）
        flip_visual_attachments = True  # 翻转视觉附件：某些.obj网格需要从y轴向上翻转到z轴向上
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0  # 回合终止惩罚
            tracking_lin_vel = 1.0  # 线速度跟踪奖励
            tracking_ang_vel = 0.5  # 角速度跟踪奖励
            lin_vel_z = -2.0  # Z轴线速度惩罚（防止跳跃）
            ang_vel_xy = -0.05  # XY轴角速度惩罚（防止翻滚）
            orientation = -0.2  # 姿态惩罚（保持直立）
            dof_acc = -2.5e-7  # 关节加速度惩罚（平滑运动）
            joint_power = -2e-5  # 关节功率惩罚（节能）
            base_height = -1.0  # 基座高度奖励（保持合适高度）
            foot_clearance = -0.01  # 足部离地高度惩罚
            action_rate = -0.01  # 动作变化率惩罚（平滑控制）
            smoothness = -0.01  # 运动平滑性惩罚
            feet_air_time =  0  # 足部悬空时间奖励（跳跃）
            collision = -0.0  # 碰撞惩罚 
            feet_stumble = -0.0  # 足部绊倒惩罚 
            stand_still = -0.0  # 静止惩罚（鼓励运动）
            torques = -0.00001  # 关节力矩惩罚 
            dof_vel = -0.0  # 关节速度惩罚
            dof_pos_limits = -0.001  # 关节位置限制惩罚 
            dof_vel_limits = 0.0  # 关节速度限制惩罚
            torque_limits = 0.0  # 力矩限制惩罚
            #new_rewards
            # hipX_limits = -0.001  # 髋关节X轴限制惩罚 -0.001
            # VHIP奖励函数
            vhip_angle = -0.1  # VHIP角度奖励（惩罚过大摆角）
            vhip_angular_acceleration = -0.001  # VHIP角加速度奖励（惩罚快速倾倒）

        only_positive_rewards = False  # 是否只保留正奖励（如果为True，负奖励会被裁剪到0，避免过早终止问题）
        tracking_sigma = 0.25  # 跟踪奖励的高斯参数：tracking_reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95  # 关节位置软限制：URDF限制的百分比，超过此值会被惩罚  0.95
        soft_dof_vel_limit = 0.95  # 关节速度软限制
        soft_torque_limit = 0.95  # 力矩软限制
        base_height_target = 0.5  # 基座目标高度 [m] 
        max_contact_force = 100.  # 最大接触力，超过此值会被惩罚
        clearance_height_target = 0.25  # 足部离地目标高度 [m] 
        # VHIP参数
        vhip_angle_threshold = 0.1  # VHIP角度阈值 [rad]
        vhip_angular_acc_threshold = 0.001  # VHIP角加速度阈值 [rad/s²]

class X30RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_X30'
        max_iterations = 8000  # number of policy updates
        # resume = True  # 继续训练
        # load_run = 'Sep08_15-57-22_'# 指定要加载的运行
        # checkpoint = 4000  # 指定要加载的检查点