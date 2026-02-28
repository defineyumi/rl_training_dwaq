#####################################
# 这个脚本用于导出带有环境编码器的策略
#####################################

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import select
import tty
import termios
import copy

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger

import numpy as np
import torch


class KeyboardController:
    """键盘控制器类，用于手动控制机器人"""
    
    def __init__(self):
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_vel = 0.0
        self.vel_step = 0.1  # 速度步长
        self.max_vel = 1.0   # 最大速度
        
        # 不再需要风格控制（只有一种风格）
        
        # 相机控制
        self.camera_mode = 0  # 0: 固定机位, 1: 后机位, 2: 左机位, 3: 右机位
        self.camera_modes = ['固定机位', '后机位', '左机位', '右机位']
        
        # 相机平滑参数
        self.camera_pos_smooth = None
        self.camera_target_smooth = None
        self.smooth_factor = 0.15  # 平滑系数，越小越平滑
        
        # 记录上次的速度命令，用于检测变化
        self.last_x_vel = None
        self.last_y_vel = None
        self.last_yaw_vel = None
        
        # 重生标志
        self.should_reset = False
        
        # 保存终端设置
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        
    def __del__(self):
        """恢复终端设置"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self):
        """获取键盘输入"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
    
    def update_commands(self):
        """根据键盘输入更新控制命令"""
        key = self.get_key()
        
        if key is None:
            return True
            
        # 前进/后退
        if key == 'w' or key == 'W':
            self.x_vel = min(self.x_vel + self.vel_step, self.max_vel)
            self.y_vel = 0.0
            self.yaw_vel = 0.0
        elif key == 's' or key == 'S':
            self.x_vel = max(self.x_vel - self.vel_step, -self.max_vel)
            self.y_vel = 0.0
            self.yaw_vel = 0.0
        # 左右移动
        elif key == 'a' or key == 'A':
            self.y_vel = min(self.y_vel + self.vel_step, self.max_vel)
            self.x_vel = 0.0
            self.yaw_vel = 0.0
        elif key == 'd' or key == 'D':
            self.y_vel = max(self.y_vel - self.vel_step, -self.max_vel)
            self.x_vel = 0.0
            self.yaw_vel = 0.0
        
        # 左右转弯
        elif key == 'q' or key == 'Q':
            self.yaw_vel = min(self.yaw_vel + self.vel_step, self.max_vel)
            self.x_vel = 0.0
            self.y_vel = 0.0
        elif key == 'e' or key == 'E':
            self.yaw_vel = max(self.yaw_vel - self.vel_step, -self.max_vel)
            self.x_vel = 0.0
            self.y_vel = 0.0
        
        # 停止
        elif key == ' ':
            self.x_vel = 0.0
            self.y_vel = 0.0
            self.yaw_vel = 0.0
        
        # 切换相机机位
        elif key == 'c' or key == 'C':
            self.camera_mode = (self.camera_mode + 1) % len(self.camera_modes)
        
        # 切换风格：按1/2/3键直接切换到对应风格
        
        # 重生机器人：按'r'键
        elif key == 'r' or key == 'R':
            self.should_reset = True
        
        # 退出
        elif key == '\x03':  # Ctrl+C
            return False
        
        return True
    
    def get_commands(self):
        """获取当前控制命令"""
        return self.x_vel, self.y_vel, self.yaw_vel
    
    
    def should_reset_robot(self):
        """检查是否需要重生机器人"""
        if self.should_reset:
            self.should_reset = False  # 重置标志
            return True
        return False
    
    def get_camera_mode(self):
        """获取当前相机模式"""
        return self.camera_mode
    
    def get_camera_follow(self):
        """获取相机跟随状态（兼容性方法）"""
        return self.camera_mode > 0
    
    def smooth_camera_position(self, target_pos, target_target):
        """平滑相机位置"""
        if self.camera_pos_smooth is None:
            # 第一次初始化
            self.camera_pos_smooth = np.array(target_pos)
            self.camera_target_smooth = np.array(target_target)
        else:
            # 使用线性插值进行平滑
            self.camera_pos_smooth = (1 - self.smooth_factor) * self.camera_pos_smooth + self.smooth_factor * np.array(target_pos)
            self.camera_target_smooth = (1 - self.smooth_factor) * self.camera_target_smooth + self.smooth_factor * np.array(target_target)
        
        return self.camera_pos_smooth, self.camera_target_smooth
    
    def print_status_if_changed(self):
        """只在速度命令改变时打印简洁的状态信息"""
        # 检查是否有变化
        if (            self.last_x_vel != self.x_vel or 
            self.last_y_vel != self.y_vel or
            self.last_yaw_vel != self.yaw_vel):
            
            # 更新记录
            self.last_x_vel = self.x_vel
            self.last_y_vel = self.y_vel
            self.last_yaw_vel = self.yaw_vel
            
            # 打印简洁的状态信息
            print(f"\r 状态: 前进{self.x_vel:+.1f} 左移{self.y_vel:+.1f} 转弯{self.yaw_vel:+.1f} 相机:{self.camera_modes[self.camera_mode]}\r", end="", flush=True)


def export_policy_as_jit(actor_critic, path):
    """导出环境编码器策略"""
    exporter = PolicyExporter(actor_critic)
    exporter.eval()
    
    # 准备测试输入（动态获取维度）
    history_size = 6
    # 从actor_critic获取单步观测维度（如果存在）
    if hasattr(actor_critic, 'num_one_step_obs'):
        one_step_obs = actor_critic.num_one_step_obs
    else:
        # 默认值（向后兼容）
        one_step_obs = 48  # 更新后的单步观测维度（包含3个风格维度）
    total_obs_dim = one_step_obs * history_size
    device = next(actor_critic.parameters()).device
    dummy_input = torch.ones(1, total_obs_dim).to(device)
    
    # 导出模型
    os.makedirs(path, exist_ok=True)
    policy_path = os.path.join(path, 'policy.pt')
    
    exporter.to('cpu')
    dummy_input_cpu = dummy_input.cpu()
    traced_script_module = torch.jit.trace(exporter, dummy_input_cpu)
    traced_script_module.save(policy_path)
    
    return policy_path

class PolicyExporter(torch.nn.Module):
    """环境编码器策略导出器 (适配新的EnvironmentActorCritic)"""
    def __init__(self, actor_critic):
        super().__init__()
        # 复制新的EnvironmentActorCritic组件
        self.actor = copy.deepcopy(actor_critic.actor)
        self.encoder = copy.deepcopy(actor_critic.encoder)
        self.encode_mean_latent = copy.deepcopy(actor_critic.encode_mean_latent)
        self.encode_logvar_latent = copy.deepcopy(actor_critic.encode_logvar_latent)
        self.encode_mean_vel = copy.deepcopy(actor_critic.encode_mean_vel)
        self.encode_logvar_vel = copy.deepcopy(actor_critic.encode_logvar_vel)
        
    def reparameterise(self, mean, logvar):
        """重参数化技巧 (推理时使用确定性输出)"""
        # 在推理时，直接使用均值，不使用随机采样
        return mean
        
    def cenet_forward(self, obs_history):
        """环境编码器前向传播"""
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        code = torch.cat((code_vel, code_latent), dim=-1)
        return code
        
    def forward(self, obs_history):
        """
        前向传播
        Args:
            obs_history: 历史观测 [batch_size, num_one_step_obs * history_size]
        Returns:
            actions: 动作 [batch_size, 12]
        """
        # 使用环境编码器获取编码
        code = self.cenet_forward(obs_history)
        
        # 动态获取单步观测维度（从历史观测维度推断）
        num_one_step_obs = obs_history.shape[1] // 6  # 假设历史长度为6
        
        # 获取当前观测（历史观测的前num_one_step_obs维）
        current_obs = obs_history[:, :num_one_step_obs]
        
        # 拼接输入：环境编码 + 当前观测
        actor_input = torch.cat((code, current_obs), dim=-1)
        
        # 通过Actor网络得到动作
        return self.actor(actor_input)

def play3(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1  # 只改为单个机器人
    env_cfg.env.episode_length_s = 1000  # 设置很长的episode时间，避免重置
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    # env_cfg.terrain.mesh_type = 'plane'
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    env.debug_viz = False    # 启用测量点可视化
    
    # 禁用episode超时重置
    env.max_episode_length = float('inf')  # 设置无限大的episode长度
    
    # 禁用接触重置机制 - 防止机器人触碰到关节时重置回合
    env.cfg.asset.terminate_after_contacts_on = []  # 清空终止接触列表
    
    # 重写check_termination方法，完全禁用接触重置
    def disable_contact_reset(self):
        """禁用接触重置，只保留超时重置"""
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf  # 只使用超时重置，不使用接触重置
    
    # 替换原有的check_termination方法
    env.check_termination = disable_contact_reset.__get__(env, env.__class__)
    
    # 初始化控制命令
    env.commands[:, 0] = 0.0  # x_vel
    env.commands[:, 1] = 0.0  # y_vel
    env.commands[:, 2] = 0.0  # yaw_vel
    # 不再需要风格维度初始化

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # 创建键盘控制器
    controller = KeyboardController()
    
    # 打印控制说明
    print("控制说明:W/S: 前进/后退,A/D: 左移/右移,Q/E: 左转/右转,空格: 停止所有运动,C: 切换相机机位,R: 重生机器人,Ctrl+C: 退出程序\r\n")
    print("相机机位: 固定机位 -> 后机位 -> 左机位 -> 右机位 -> 固定机位...\r\n")

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'environment_encoder')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    try:
        i = 0
        while True:  # 无限循环，直到用户按Ctrl+C退出
            # 更新控制命令
            if not controller.update_commands():
                break
            
            # 检查是否需要重生机器人
            if controller.should_reset_robot():
                # 重置第一个环境（索引0）
                env.reset_idx(torch.tensor([0], device=env.device))
                # 重新获取观测
                obs = env.get_observations()
                
            # 获取当前控制命令
            x_vel, y_vel, yaw_vel = controller.get_commands()
        
            actions = policy(obs.detach())
            env.commands[:, 0] = x_vel
            env.commands[:, 1] = y_vel
            env.commands[:, 2] = yaw_vel
            obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            # 相机控制
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)
            elif controller.get_camera_follow():
                # 多机位相机跟随机器人
                robot_pos = env.root_states[0, 0:3].cpu().numpy()  # 机器人位置
                robot_quat = env.root_states[0, 3:7].cpu().numpy()  # 机器人四元数
                
                # 从四元数计算旋转矩阵
                from scipy.spatial.transform import Rotation
                # Isaac Gym四元数格式是[x,y,z,w]，scipy需要[x,y,z,w]格式
                r = Rotation.from_quat([robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]])  # x,y,z,w
                forward_vec = r.apply([1, 0, 0])  # 机器人前进方向
                right_vec = r.apply([0, 1, 0])    # 机器人右侧方向
                up_vec = r.apply([0, 0, 1])       # 机器人上方方向
                
                # 根据相机模式计算相机位置
                camera_mode = controller.get_camera_mode()
                offset_distance = 3.0  # 距离机器人的距离
                offset_height = 1.0     # 相机高度（降低以便更好地观察机器人）
                
                if camera_mode == 1:  # 后机位
                    target_camera_pos = robot_pos - forward_vec * offset_distance + [0, 0, offset_height]
                    target_camera_target = robot_pos + [0, 0, 0.5]
                elif camera_mode == 2:  # 左机位
                    target_camera_pos = robot_pos - right_vec * offset_distance + [0, 0, offset_height]
                    target_camera_target = robot_pos + [0, 0, 0.5]
                elif camera_mode == 3:  # 右机位
                    target_camera_pos = robot_pos + right_vec * offset_distance + [0, 0, offset_height]
                    target_camera_target = robot_pos + [0, 0, 0.5]
                
                # 使用平滑相机位置
                smooth_camera_pos, smooth_camera_target = controller.smooth_camera_position(target_camera_pos, target_camera_target)
                
                env.set_camera(smooth_camera_pos, smooth_camera_target)

            # 显示美观的状态信息（只在变化时显示）
            controller.print_status_if_changed()
            
            i += 1  # 增加步数
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理资源
        del controller
        print("程序结束")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play3(args)