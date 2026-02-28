from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Lite3DwaqRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096  # 环境数量
        num_one_step_observations = 45  # 单步观测维度（3维运动指令+3维基座角速度+3维重力+12维关节位置+12维关节速度+12维动作）
        use_history_obs = True # 是否使用历史信息
        # 根据开关动态设置观测维度
        if use_history_obs:
            num_observations = num_one_step_observations * 6  # 270维（45*6）
        else:
            num_observations = num_one_step_observations      # 45维
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # 单步特权观测维度（45维单步观测+3维基座线速度+3维外部力+187维扫描点）
        num_privileged_obs = num_one_step_privileged_obs  # 总特权观测维度（只使用单步特权观测）
        num_actions = 12  # 动作维度
        env_spacing = 3.  # 环境间距（不用于高度场/三角网格）
        send_timeouts = True  # 向算法发送超时信息
        episode_length_s = 20  # 回合长度（秒）

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # 网格类型：none, plane, heightfield 或 trimesh
        horizontal_scale = 0.1 # [m] 水平缩放比例
        vertical_scale = 0.005 # [m] 垂直缩放比例
        border_size = 25 # [m] 边界大小
        curriculum = True # 是否启用课程学习
        static_friction = 1.0 # 静摩擦系数
        dynamic_friction = 1.0 # 动摩擦系数
        restitution = 0. # 恢复系数
        # 粗糙地形专用：
        measure_heights = True # 是否测量高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # X方向测量点 [m] - 1m x 1.6m 矩形（不含中心线）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # Y方向测量点 [m]
        selected = False # 是否选择唯一地形类型并传递所有参数
        terrain_kwargs = None # 所选地形的参数字典
        max_init_terrain_level = 0 # 初始课程学习状态（起始难度等级）
        terrain_length = 8. # 地形长度 [m]
        terrain_width = 8. # 地形宽度 [m]
        num_rows= 10 # 地形行数（难度等级数）
        num_cols = 20 # 地形列数（地形类型数）
        # 地形类型：[平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散障碍]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1] # 各地形类型的比例
        # 仅适用于 trimesh：
        slope_treshold = 0.75 # 坡度阈值，超过此阈值的斜坡将被修正为垂直表面

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 2.0 #最大难度 - 限制最大速度为2m/s
        num_commands = 4 # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s] - 速度范围，主要用于站立测试
            lin_vel_y = [-1.0, 1.0]   # min max [m/s] - 横向速度范围
            ang_vel_yaw = [-3.14 , 3.14]   # min max [rad/s] - 角速度范围
            heading = [-3.14, 3.14] # 航向角范围 [rad]（最小值，最大值）


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.27] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.0,   # [rad] 顺时针为正
            'HL_HipX_joint': 0.0,   # [rad]
            'FR_HipX_joint': 0.0,   # [rad]
            'HR_HipX_joint': 0.0,   # [rad]

            'FL_HipY_joint': -0.8,   # [rad] 逆时针为正
            'HL_HipY_joint': -0.8,   # [rad] 
            'FR_HipY_joint': -0.8,   # [rad] 
            'HR_HipY_joint': -0.8,   # [rad] 

            'FL_Knee_joint': 1.8,   # [rad] 顺时正为正
            'HL_Knee_joint': 1.8,   # [rad] 
            'FR_Knee_joint': 1.8,   # [rad] 
            'HR_Knee_joint': 1.8,   # [rad] 
        }


    class control( LeggedRobotCfg.control ):
        # PD控制器参数:
        control_type = 'P'  # 控制类型：P表示比例控制（只学习应该到达的角度，不学习到达这个角度需要多大的力）
        # 不同关节使用不同的KP值：
        stiffness = {'HipX': 40.0, 'HipY': 40.0, 'Knee': 40.0}  # 关节刚度 [N*m/rad]
        damping = {'HipX': 1.0, 'HipY': 1.0, 'Knee': 1.0}     # 关节阻尼 [N*m*s/rad]
        # 动作缩放：目标角度 = actionScale * action + defaultAngle
        action_scale = 0.25  # 动作缩放系数
        # 抽取：每个策略时间步长内仿真时间步长的控制动作更新次数
        decimation = 4  # 控制频率抽取因子
        hip_reduction = 1.0  # 髋关节减速比

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
        name = "lite3"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK","TORSO"]  # 接触时会被惩罚的部件（大腿、小腿）
        terminate_after_contacts_on = ["TORSO"]  # 接触时会终止回合的部件（躯干）
        privileged_contacts_on = ["TORSO", "THIGH", "SHANK"]  # 特权观测中包含接触信息的部件
        self_collisions = 1  # 自碰撞检测：1表示禁用，0表示启用（位掩码过滤器）
        flip_visual_attachments = False  # 翻转视觉附件：某些.obj网格需要从y轴向上翻转到z轴向上
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # ========== 通用奖励 ==========
            # 基础运动控制
            termination = -0.0  # 回合终止惩罚
            tracking_lin_vel = 1.0  # 线速度跟踪奖励
            tracking_ang_vel = 0.5  # 角速度跟踪奖励
            lin_vel_z = -2.0  # Z轴线速度惩罚
            ang_vel_xy = -0.05  # XY轴角速度惩罚（防止翻滚）
            orientation = -0.2  # 姿态惩罚（保持直立）
            foot_clearance = -0.01  # 足部离地高度惩罚

            # 站立静止相关奖励
            base_height = -1.0  # 基座高度奖励（保持合适高度）
            stand_still_pose = 1.0  # 静止站立姿态奖励：当速度<0.15时，惩罚偏离初始姿态（根据风格选择参考姿态）

            # 运动平滑性
            dof_acc = -2.5e-7  # 关节加速度惩罚（平滑运动）
            joint_power = -2e-5  # 关节功率惩罚（节能）
            action_rate = -0.01  # 动作变化率惩罚（平滑控制）
            smoothness = -0.01  # 运动平滑性惩罚
            torques = -1e-5  # 关节力矩惩罚   
            
            # VHIP奖励函数
            vhip_angle = -0.1  # VHIP角度奖励（惩罚过大摆角）
            vhip_angular_acceleration = -0.001  # VHIP角加速度奖励（惩罚快速倾倒）        
 
            # ========== 未使用的奖励 ==========
            joint_pos_tracking = -0.0  # 关节位置跟踪奖励（保持初始姿态）必须是负 -0.03
            feet_stumble = -0.0  # 足部绊倒惩罚
            feet_air_time = 0  # 足部悬空时间奖励
            collision = -0.0  # 碰撞惩罚  
            stand_still = -0.0  # 静止惩罚
            dof_vel = -0.0  # 关节速度惩罚
            dof_pos_limits = 0.0  # 关节位置限制惩罚 
            dof_vel_limits = 0.0  # 关节速度限制惩罚
            torque_limits = 0.0  # 力矩限制惩罚

        only_positive_rewards = False  # 是否只保留正奖励（如果为True，负奖励会被裁剪到0，避免过早终止问题）
        tracking_sigma = 0.25  # 跟踪奖励的高斯参数：tracking_reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95  # 关节位置软限制：URDF限制的百分比，超过此值会被惩罚
        soft_dof_vel_limit = 0.95  # 关节速度软限制
        soft_torque_limit = 0.95  # 力矩软限制
        base_height_target_normal = 0.27  # 正常站立时的基座目标高度 [m]
        max_contact_force = 100.  # 最大接触力，超过此值会被惩罚
        clearance_height_target = 0.02  # 摆动腿足部离地目标高度 [m]
        # VHIP参数
        vhip_angle_threshold = 0.1  # VHIP角度阈值 [rad]
        vhip_angular_acc_threshold = 0.001  # VHIP角加速度阈值 [rad/s²]

class Lite3DwaqRoughCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'EnvironmentOnPolicyRunner'  # 使用环境编码器训练运行器
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # 动态熵系数配置
        use_dynamic_entropy = True  # 是否启用基于性能的动态熵系数
        entropy_coef = 0.01  # 基础熵系数
        
        # 性能指标阈值（基于差距递减式计算熵系数）
        lin_vel_threshold = 0.7  # 线速度跟踪目标值
        ang_vel_threshold = 0.3  # 角速度跟踪目标值
        terrain_level_threshold = 5  # 地形等级目标值
        
        # 熵系数范围
        high_entropy_coef = 0.02  # 高熵系数（训练初期）
        low_entropy_coef = 0.01   # 低熵系数（策略成熟后）
        
        # Softmax温度参数
        softmax_temperature = 2.0  # 控制softmax的锐度，值越大越平滑
        
        # 环境编码器参数
        env_encoder_lr = 1e-3  # 环境编码器学习率
        env_encoder_beta = 1.0  # KL散度权重（过大会导致编码器坍塌）
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'EnvironmentActorCritic'
        algorithm_class_name = 'EnvironmentPPO'

        run_name = 'Dwaq'
        experiment_name = 'rough_lite3'
        max_iterations = 4000  # number of policy updates
        # resume = True  # 继续训练
        # load_run = 'Dec02_10-29-12_Dwaq'# 指定要加载的运行
        # checkpoint = 3500  # 指定要加载的检查点