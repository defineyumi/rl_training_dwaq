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

def split_and_pad_trajectories(tensor, dones):
    """ 
    在完成标志处分割轨迹，然后拼接并用零填充至最长轨迹长度。
    返回与轨迹有效部分对应的掩码。
    
    示例: 
        输入: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        输出:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    假设输入张量维度顺序为: [时间步, 环境数量, 其他维度]
    """
    # 克隆done标志以避免修改原始数据
    dones = dones.clone()
    # 确保最后一个时间步标记为完成，以便正确处理最后一段轨迹
    dones[-1] = 1
    
    # 将dones转置并展平，以便按环境顺序处理
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # 找到所有完成标志的位置索引
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    # 计算每段轨迹的长度（相邻完成标志之间的间隔）
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    # 将轨迹长度转换为列表形式
    trajectory_lengths_list = trajectory_lengths.tolist()
    
    # 转置并展平输入张量，然后按轨迹长度分割
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # 使用pad_sequence将所有轨迹填充到相同长度（最长轨迹的长度）
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    # 创建掩码，标识轨迹中的有效部分（非填充部分）
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    
    # 返回填充后的轨迹和对应的掩码
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ 
    执行split_and_pad_trajectories()的逆操作，从填充的轨迹中恢复原始数据
    """
    # 转置掩码和轨迹，应用掩码选择有效数据，然后重新调整形状并转置回原始格式
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)