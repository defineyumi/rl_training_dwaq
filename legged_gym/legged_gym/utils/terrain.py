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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        """
        地形类初始化，负责创建和管理多环境地形网格
        
        Args:
            cfg: 地形配置参数，包含网格大小、地形类型、缓冲区等设置
            num_robots: 机器人数量，用于多机器人训练
        """
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        
        # 如果地形类型为none或plane，直接返回（不需要生成复杂地形）
        if self.type in ["none", 'plane']:
            return
            
        # 设置单个环境的物理尺寸
        self.env_length = cfg.terrain_length      # 环境长度（米）
        self.env_width = cfg.terrain_width       # 环境宽度（米）
        
        # 计算地形类型的累积分布比例，用于随机选择地形类型
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # 计算总子地形数量（网格总数）
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # 初始化每个环境的原点坐标（x, y, z）
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # 计算每个环境在像素坐标系中的尺寸
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)      # 环境宽度（像素）
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)    # 环境长度（像素）

        # 计算缓冲区在像素坐标系中的大小
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # 计算总地形数组的尺寸（包含所有环境和缓冲区）
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border    # 总列数
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border   # 总行数

        # 初始化高度场数组，存储整个地形的像素级高度数据
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        # 根据配置选择地形生成模式
        if cfg.curriculum:
            self.curiculum()          # 课程学习模式：难度递增
        elif cfg.selected:
            self.selected_terrain()  # 选择模式：特定地形类型
        else:    
            self.randomized_terrain() # 随机模式：随机地形和难度
        
        # 设置高度采样数据
        self.heightsamples = self.height_field_raw
        
        # 如果使用三角网格，将高度场转换为三角网格数据
        if self.type=="trimesh":
            self.vertices, self.triangles = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        """
        随机地形生成模式：为每个环境格子随机生成地形类型和难度
        适合探索性训练，提供多样化的训练环境
        """
        for k in range(self.cfg.num_sub_terrains):
            # 将一维索引转换为二维网格坐标
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 随机选择地形类型（0-1之间的值）
            choice = np.random.uniform(0, 1)
            # 随机选择难度等级（0.5, 0.75, 0.9三个等级）
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # 生成地形并添加到地图中
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        """
        课程学习模式：按难度递增和地形类型变化生成地形
        行方向：难度从0到1递增（从简单到困难）
        列方向：地形类型从0到1变化（从平滑到复杂）
        实现渐进式训练，让机器人逐步适应更困难的地形
        """
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                # 行方向：难度递增（0到1）
                difficulty = i / self.cfg.num_rows
                # 列方向：地形类型变化（0到1，加0.001避免边界问题）
                choice = j / self.cfg.num_cols + 0.001

                # 生成地形并添加到地图中
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        """
        选择模式：生成特定类型的地形
        用于专项训练，所有环境都使用相同的地形类型和参数
        适合针对特定地形进行强化训练
        """
        # 从配置中获取指定的地形类型
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # 将一维索引转换为二维网格坐标
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 创建子地形对象
            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,      # 地形宽度（像素）
                              length=self.width_per_env_pixels,     # 地形长度（像素）
                              vertical_scale=self.vertical_scale,   # 垂直缩放比例
                              horizontal_scale=self.horizontal_scale) # 水平缩放比例

            # 使用eval动态调用指定的地形生成函数
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            # 将生成的地形添加到地图中
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        """
        根据选择的地形类型和难度参数生成地形
        
        Args:
            choice (float): 地形类型选择值，用于确定生成哪种地形
            difficulty (float): 难度参数，影响地形的各种参数（如坡度、高度等）
        """
        # 创建子地形对象，用于构建单个环境的地形
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,      # 地形宽度（像素）
                                length=self.width_per_env_pixels,     # 地形长度（像素）
                                vertical_scale=self.cfg.vertical_scale,    # 垂直缩放比例
                                horizontal_scale=self.cfg.horizontal_scale) # 水平缩放比例
        
        # 根据难度参数计算各种地形参数
        slope = difficulty * 0.4                    # 坡度：难度越高，坡度越大
        amplitude = 0.01 + 0.07 * difficulty       # 振幅：用于随机地形的高度变化范围
        step_height = 0.05 + 0.14 * difficulty     # 台阶高度：难度越高，台阶越高
        discrete_obstacles_height = 0.05 + difficulty * 0.1  # 离散障碍物高度
        stepping_stones_size = 1.5 * (1.05 - difficulty)     # 踏脚石大小：难度越高，石头越小
        stone_distance = 0.05 if difficulty==0 else 0.1      # 踏脚石间距：难度为0时较小，否则较大
        gap_size = 1. * difficulty                 # 间隙大小：难度越高，间隙越大
        pit_depth = 1. * difficulty                # 坑深度：难度越高，坑越深
        
        # 根据choice值选择地形类型
        if choice < self.proportions[0]:
            # 平滑斜坡地形
            if choice < self.proportions[0]/ 2:
                slope *= -1  # 前半部分为下坡，后半部分为上坡
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            
        elif choice < self.proportions[1]:
            # 粗糙斜坡地形：在平滑斜坡基础上添加随机噪声
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
            
        elif choice < self.proportions[3]:
            # 楼梯地形（包括上楼梯和下楼梯）
            if choice<self.proportions[2]:
                step_height *= -1  # 前半部分为下楼梯，后半部分为上楼梯
                
            wide_pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3.)
            
        elif choice < self.proportions[4]:
            # 离散障碍物地形：随机分布的矩形障碍物
            num_rectangles = 20                     # 障碍物数量
            rectangle_min_size = 1.                 # 最小障碍物尺寸
            rectangle_max_size = 2.                 # 最大障碍物尺寸
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)

        elif choice < self.proportions[5]:
            # 间隙地形：需要跨越的沟壑
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)

        elif choice < self.proportions[6]:
            # 踏脚石地形：需要跳跃的石头阵列
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
            
        else:
            # 坑地形：需要避开的深坑
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        """
        将生成的地形添加到总地形地图中的指定位置
        
        Args:
            terrain: 子地形对象，包含该环境的地形数据
            row: 行索引（i坐标）
            col: 列索引（j坐标）
        """
        i = row
        j = col
        
        # 计算在总地形数组中的像素坐标范围
        start_x = self.border + i * self.length_per_env_pixels      # 起始X坐标（包含缓冲区）
        end_x = self.border + (i + 1) * self.length_per_env_pixels  # 结束X坐标
        start_y = self.border + j * self.width_per_env_pixels        # 起始Y坐标（包含缓冲区）
        end_y = self.border + (j + 1) * self.width_per_env_pixels   # 结束Y坐标
        
        # 将子地形的高度数据复制到总地形数组中
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # 计算环境在世界坐标系中的原点位置
        env_origin_x = (i + 0.5) * self.env_length    # X坐标：环境中心
        env_origin_y = (j + 0.5) * self.env_width     # Y坐标：环境中心
        
        # 计算环境中心区域的高度范围（用于确定Z坐标）
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)  # 中心区域起始X
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)  # 中心区域结束X
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)   # 中心区域起始Y
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)   # 中心区域结束Y
        
        # 计算环境中心区域的最大高度作为环境原点Z坐标
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        # 存储环境原点坐标
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:
    
    If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
               B(x2,y2)
              /|
             / |
            /  |
    (x1,y1)A---A'(x2',y1)
    
    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]
    
    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale
    else:
        move_x = np.zeros((num_rows, num_cols))
    
    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3
    
    return vertices, triangles

def gap_terrain(terrain, gap_size, platform_size=1.):
    """
    生成间隙地形：在地形中心创建一个需要跨越的沟壑
    机器人需要跳跃或跨越这个间隙才能通过
    
    Args:
        terrain: 子地形对象
        gap_size: 间隙大小（米）
        platform_size: 平台大小（米）
    """
    # 将物理尺寸转换为像素尺寸
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    # 计算地形中心坐标
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    
    # 计算间隙和平台的像素坐标范围
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    # 创建间隙：外圈设为深坑（-1000表示很深的坑）
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    # 内圈设为平地（0表示地面高度）
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def flat_terrain(terrain, platform_size=1.):
    """
    生成平坦地形：整个地形都是平坦的，高度为0
    用于基础训练和测试
    
    Args:
        terrain: 子地形对象
        platform_size: 平台大小（米，此函数中未使用）
    """
    terrain.height_field_raw[:, :] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    """
    生成坑地形：在地形中心创建一个深坑
    机器人需要避开这个坑，或者需要从坑中爬出
    
    Args:
        terrain: 子地形对象
        depth: 坑的深度（米）
        platform_size: 平台大小（米）
    """
    # 将物理尺寸转换为像素尺寸
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    
    # 计算坑的像素坐标范围
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    
    # 在中心区域创建深坑
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def wide_pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    """
    加宽第一层台阶的台阶生成函数，后续台阶宽度不变

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    current_step_width = step_width
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        if height==0:
            current_step_width = step_width * 4
        else:
            current_step_width = step_width
        start_x += current_step_width
        stop_x -= current_step_width
        start_y += current_step_width
        stop_y -= current_step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain