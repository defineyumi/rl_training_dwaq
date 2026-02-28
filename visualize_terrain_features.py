#!/usr/bin/env python3
"""
地形特征可视化工具
使用t-SNE将高维地形特征降维到2D进行可视化
"""

import os
import sys
import argparse
from datetime import datetime

# 添加路径
sys.path.append('/home/cjl/rl_training/legged_gym')
sys.path.append('/home/cjl/rl_training/rsl_rl')

# 关键：必须先导入isaacgym，再导入torch和其他库
import isaacgym
from isaacgym import gymapi

# 现在可以导入torch和其他库了
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']  # 使用系统字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

class TerrainFeatureVisualizer:
    """地形特征可视化器"""
    
    def __init__(self, model_path, task_name=None, num_samples=2000, save_dir=None):
        """
        初始化地形特征可视化器
        
        Args:
            model_path: 模型路径
            task_name: 任务名称（可选，会自动检测）
            num_samples: 采样数量
                - 最少: 500 (快速预览)
                - 推荐: 2000-3000 (平衡准确性和速度)
                - 最佳: 5000+ (最准确，但计算时间长)
            save_dir: 保存目录（可选）
        """
        self.model_path = model_path
        
        # 自动检测任务名称（从模型路径中提取）
        if task_name is None:
            # 尝试从路径中提取任务名称
            if 'lite3' in model_path.lower():
                self.task_name = 'lite3'
            elif 'x30' in model_path.lower():
                self.task_name = 'x30'
            elif 'aliengo' in model_path.lower():
                self.task_name = 'aliengo'
            else:
                self.task_name = 'lite3'  # 默认
                print(f"警告: 无法从路径推断任务名称，使用默认值: {self.task_name}")
        else:
            self.task_name = task_name
        
        self.num_samples = num_samples
        
        # 创建保存目录（在模型所在目录下）
        if save_dir is None:
            model_dir = os.path.dirname(os.path.abspath(model_path))
            self.save_dir = os.path.join(model_dir, f"terrain_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("=" * 60)
        print("地形特征可视化器")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"任务名称: {self.task_name} (自动检测)")
        print(f"采样数量: {num_samples}")
        print(f"结果保存到: {self.save_dir}")
        print("=" * 60)
    
    def load_model_and_env(self):
        """加载模型和环境"""
        print("正在加载模型和环境...")
        
        class SimpleArgs:
            def __init__(self, task_name):
                self.task = task_name
                self.headless = True
                self.resume = True
                self.sim_device = 'cuda:0'
                self.pipeline = 'gpu'
                self.graphics_device_id = 0
                self.flex = False
                self.physx = True
                self.num_threads = 0
                self.subscenes = 0
                self.slices = 0
                self.horovod = False
                self.rl_device = 'cuda:0'
                self.num_envs = 4096  # 使用更多环境以快速收集数据
                self.seed = 1
                self.max_iterations = 1
                self.experiment_name = 'rough_lite3'
                self.run_name = ''
                self.load_run = ''
                self.checkpoint = -1
                self.resume = False
                self.physics_engine = gymapi.SIM_PHYSX
                self.use_gpu = True
                self.use_gpu_pipeline = True
                self.num_gpu = 1
                self.num_subscenes = 0
                self.sim_device_type = 'cuda'
                self.rl_device_type = 'cuda'
                self.device = 'cuda:0'
        
        # 检查模型文件，判断是否使用环境编码器
        # 如果模型包含encoder相关参数，说明使用了EnvironmentActorCritic
        use_env_encoder = False
        task_name_to_use = self.task_name
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model_state = checkpoint.get('model_state_dict', {})
            has_env_encoder = any('encoder' in key or 'decoder' in key for key in model_state.keys())
            
            if has_env_encoder:
                use_env_encoder = True
                print("检测到模型包含环境编码器，将使用 EnvironmentActorCritic")
                # 如果路径包含Dwaq，使用lite3_dwaq配置（注意是小写）
                if 'dwaq' in self.model_path.lower():
                    print("检测到Dwaq模型，使用 lite3_dwaq 配置")
                    task_name_to_use = 'lite3_dwaq'
        except Exception as e:
            print(f"警告: 无法检查模型类型: {e}")
            # 如果路径包含Dwaq，尝试使用lite3_dwaq
            if 'dwaq' in self.model_path.lower():
                print("从路径检测到Dwaq，尝试使用 lite3_dwaq 配置")
                task_name_to_use = 'lite3_dwaq'
                use_env_encoder = True
        
        args = SimpleArgs(task_name_to_use)
        
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        _, train_cfg = task_registry.get_cfgs(name=args.task)
        train_cfg.runner.resume = False
        train_cfg.runner.load_run = ''
        train_cfg.runner.checkpoint = -1
        
        # 如果检测到环境编码器，确保配置正确
        if use_env_encoder:
            # 确保使用正确的runner类
            train_cfg.runner_class_name = 'EnvironmentOnPolicyRunner'
            train_cfg.runner.policy_class_name = 'EnvironmentActorCritic'
            train_cfg.runner.algorithm_class_name = 'EnvironmentPPO'
        
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        
        if os.path.exists(self.model_path):
            ppo_runner.load(self.model_path)
            print(f"成功加载模型: {self.model_path}")
        else:
            print(f"模型文件不存在: {self.model_path}")
            return None, None
        
        return env, ppo_runner
    
    def collect_terrain_features(self, env, ppo_runner):
        """收集地形特征数据"""
        print("\n正在收集地形特征数据...")
        print(f"目标收集 {self.num_samples} 个样本")
        
        actor_critic = ppo_runner.alg.actor_critic
        
        if not hasattr(actor_critic, 'get_env_features'):
            print("错误: 模型中没有环境编码器 (get_env_features 方法)")
            return None, None, None, None
        
        terrain_features = []
        
        # 重置环境
        obs = env.get_observations()
        
        # 收集数据
        collected = 0
        episode_counter = 0
        step_count = 0
        
        print("开始收集数据（这可能需要几分钟）...")
        
        while collected < self.num_samples:
            step_count += 1
            
            # 获取动作
            with torch.no_grad():
                actions = actor_critic.act_inference(obs)
            
            # 环境步进
            obs, _, _, dones, infos, _, _ = env.step(actions)
            
            # 提取地形特征
            with torch.no_grad():
                # 获取历史观测（270维）
                obs_hist = obs  # 当前观测应该包含历史信息
                
                # 检查观测维度
                if obs_hist.shape[1] == 270:
                    # 正确维度，提取特征
                    env_speed, env_latent = actor_critic.get_env_features(obs_hist)
                    
                    # 调试：检查第一次提取的特征
                    if step_count == 1 and collected == 0:
                        print(f"\n  第一次特征提取调试信息:")
                        print(f"    obs_hist shape: {obs_hist.shape}")
                        print(f"    obs_hist range: [{obs_hist.min().item():.4f}, {obs_hist.max().item():.4f}]")
                        print(f"    env_latent shape: {env_latent.shape}")
                        print(f"    env_latent dtype: {env_latent.dtype}")
                        print(f"    env_latent range: [{env_latent.min().item():.6f}, {env_latent.max().item():.6f}]")
                        print(f"    env_latent mean: {env_latent.mean().item():.6f}")
                        print(f"    env_latent std: {env_latent.std().item():.6f}")
                        print(f"    env_latent sample (first env, all dims): {env_latent[0].cpu().numpy()}")
                        print(f"    env_latent sample (all envs, first dim): {env_latent[:, 0].cpu().numpy()}")
                    
                    # 收集特征
                    batch_size = env_latent.shape[0]
                    env_latent_np = env_latent.cpu().numpy()
                    
                    # 检查是否有NaN或Inf
                    if np.isnan(env_latent_np).any():
                        print(f"  ⚠️  警告: 检测到NaN值在步骤{step_count}")
                    if np.isinf(env_latent_np).any():
                        print(f"  ⚠️  警告: 检测到Inf值在步骤{step_count}")
                    
                    terrain_features.append(env_latent_np)
                    
                    collected += batch_size
                elif obs_hist.shape[1] == 45:
                    # 只有当前观测，需要构建历史观测
                    # 这里简化处理：重复当前观测6次
                    obs_hist_expanded = obs_hist.repeat(1, 6).reshape(obs_hist.shape[0], 270)
                    env_speed, env_latent = actor_critic.get_env_features(obs_hist_expanded)
                    
                    batch_size = env_latent.shape[0]
                    terrain_features.append(env_latent.cpu().numpy())
                    
                    collected += batch_size
                else:
                    # 未知维度，跳过
                    continue
            
            
            # 进度提示
            if step_count % 10 == 0:
                print(f"  已收集 {collected}/{self.num_samples} 个样本 (进度: {collected/self.num_samples*100:.1f}%)")
        
        # 转换为numpy数组
        if len(terrain_features) == 0:
            print("错误: 未能收集到任何特征数据")
            return None
        
        terrain_features = np.concatenate(terrain_features, axis=0)[:self.num_samples]
        
        print(f"\n数据收集完成!")
        print(f"  - 地形特征形状: {terrain_features.shape}")
        print(f"  - 收集样本数: {len(terrain_features)}")
        
        # 详细检查特征数据
        print(f"\n特征数据详细检查:")
        print(f"  - 特征值范围: [{terrain_features.min():.6f}, {terrain_features.max():.6f}]")
        print(f"  - 特征均值: {terrain_features.mean():.6f}")
        print(f"  - 特征标准差: {terrain_features.std():.6f}")
        print(f"  - 各维度均值: {np.mean(terrain_features, axis=0)}")
        print(f"  - 各维度标准差: {np.std(terrain_features, axis=0)}")
        
        # 检查是否有重复数据
        unique_features = np.unique(terrain_features, axis=0)
        print(f"  - 唯一特征数量: {len(unique_features)}/{len(terrain_features)}")
        if len(unique_features) < len(terrain_features) * 0.1:
            print(f"  ⚠️  警告: 大量重复特征！只有{len(unique_features)}个唯一特征")
        
        # 检查前几个样本
        print(f"\n前5个样本的特征值:")
        for i in range(min(5, len(terrain_features))):
            print(f"  样本{i}: {terrain_features[i]}")
        
        return terrain_features
    
    def visualize_with_tsne(self, terrain_features):
        """使用t-SNE可视化地形特征"""
        print("正在使用t-SNE降维...")
        
        # t-SNE参数
        perplexity = min(30, self.num_samples // 4)  # perplexity不能大于样本数
        if perplexity < 5:
            perplexity = 5
        
        print(f"t-SNE参数: perplexity={perplexity}, n_samples={len(terrain_features)}")
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   n_iter=1000, verbose=1)
        features_2d = tsne.fit_transform(terrain_features)
        
        print("t-SNE降维完成！")
        
        # 创建可视化
        fig = plt.figure(figsize=(14, 10))
        
        # 添加整体标题和说明
        fig.suptitle('Terrain Feature Visualization using t-SNE with K-Means Clustering', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 添加说明文字
        explanation_text = (
            "t-SNE reduces 16-dimensional terrain features to 2D for visualization.\n"
            "Points are colored by K-means clusters. If similar terrains cluster together, the encoder learned good representations."
        )
        fig.text(0.5, 0.94, explanation_text, ha='center', fontsize=11, 
                style='italic', color='gray')
        
        # 创建单个子图
        ax = plt.subplot(1, 1, 1)
        
        # 使用K-means聚类着色
        try:
            from sklearn.cluster import KMeans
            
            # 使用肘部法则确定最优聚类数
            print("正在使用肘部法则确定最优聚类数...")
            max_k = min(15, max(10, len(features_2d) // 100))  # 最大测试k值，根据样本数调整
            min_k = 2  # 最小k值
            k_range = range(min_k, max_k + 1)
            
            inertias = []
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_temp.fit(terrain_features)
                inertias.append(kmeans_temp.inertia_)
            
            # 计算肘部点：找到WCSS下降率变化最大的点
            # 使用二阶导数（变化率的变化率）来找到肘部
            if len(inertias) >= 3:
                # 计算一阶导数（变化率）
                first_diff = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                # 计算二阶导数（变化率的变化率）
                second_diff = [first_diff[i] - first_diff[i+1] for i in range(len(first_diff)-1)]
                # 找到二阶导数最大的点（变化率变化最大的地方）
                if len(second_diff) > 0:
                    elbow_idx = np.argmax(second_diff) + 1  # +1因为second_diff比k_range少2个元素
                    n_clusters = k_range[elbow_idx]
                else:
                    # 如果计算失败，使用默认值
                    n_clusters = min(10, max(5, len(features_2d) // 100))
            else:
                # 如果k值太少，使用默认值
                n_clusters = min(10, max(5, len(features_2d) // 100))
            
            print(f"肘部法则确定的最优聚类数: {n_clusters}")
            print(f"  测试范围: {min_k} 到 {max_k}")
            print(f"  各k值的WCSS: {[f'k={k}: {inertia:.2f}' for k, inertia in zip(k_range, inertias)]}")
            
            # 使用最优聚类数进行最终聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(terrain_features)
            
            # 使用鲜艳的颜色
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            for i in range(n_clusters):
                mask = cluster_labels == i
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                         alpha=0.85, s=30, c=[colors[i]], 
                         label=f'Cluster {i}', edgecolors='white', linewidths=0.8)
            
            ax.set_title('Terrain Features Colored by K-Means Clusters', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.legend(loc='best', fontsize=11, framealpha=0.95, ncol=2, 
                     markerscale=1.5, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.text(0.02, 0.98, f'Clusters: {n_clusters}, Total points: {len(features_2d)}', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 打印聚类统计信息
            print(f"\nK-means聚类结果:")
            for i in range(n_clusters):
                count = np.sum(cluster_labels == i)
                print(f"  Cluster {i}: {count} 个样本 ({count/len(cluster_labels)*100:.1f}%)")
                
        except ImportError:
            print("错误: 需要安装sklearn才能使用K-means聚类")
            ax.scatter(features_2d[:, 0], features_2d[:, 1],
                      alpha=0.6, s=20, c='steelblue', edgecolors='none')
            ax.set_title('Terrain Features (Uniform Color)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
        plt.savefig(f'{self.save_dir}/terrain_features_tsne.png', dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {self.save_dir}/terrain_features_tsne.png")
        plt.close()
        
        # 保存降维后的数据
        np.save(f'{self.save_dir}/features_2d.npy', features_2d)
        np.save(f'{self.save_dir}/terrain_features.npy', terrain_features)
        
        return features_2d
    
    def visualize_feature_statistics(self, terrain_features):
        """可视化特征统计信息"""
        print("正在生成特征统计图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Terrain Feature Statistics', fontsize=16, fontweight='bold')
        
        # 1. 特征均值
        feature_means = np.mean(terrain_features, axis=0)
        axes[0, 0].bar(range(len(feature_means)), feature_means, 
                      color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Feature Mean Values (16 dimensions)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Feature Dimension Index', fontsize=10)
        axes[0, 0].set_ylabel('Mean Value', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].text(0.02, 0.98, 'Shows average value of each feature dimension', 
                        transform=axes[0, 0].transAxes, fontsize=8, 
                        verticalalignment='top', style='italic', color='gray')
        
        # 2. 特征标准差
        feature_stds = np.std(terrain_features, axis=0)
        axes[0, 1].bar(range(len(feature_stds)), feature_stds,
                      color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Feature Standard Deviations (16 dimensions)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Feature Dimension Index', fontsize=10)
        axes[0, 1].set_ylabel('Standard Deviation', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].text(0.02, 0.98, 'Higher std = more variation in this dimension', 
                        transform=axes[0, 1].transAxes, fontsize=8, 
                        verticalalignment='top', style='italic', color='gray')
        
        # 3. 特征分布直方图
        axes[1, 0].hist(terrain_features.flatten(), bins=50, 
                       color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Overall Feature Value Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Feature Value', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].text(0.02, 0.98, 'Distribution of all feature values across all samples', 
                        transform=axes[1, 0].transAxes, fontsize=8, 
                        verticalalignment='top', style='italic', color='gray')
        
        # 4. 特征相关性热图（如果维度不太多）
        if terrain_features.shape[1] <= 20:
            corr_matrix = np.corrcoef(terrain_features.T)
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', 
                                 vmin=-1, vmax=1, aspect='auto')
            axes[1, 1].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Feature Dimension Index', fontsize=10)
            axes[1, 1].set_ylabel('Feature Dimension Index', fontsize=10)
            cbar = plt.colorbar(im, ax=axes[1, 1])
            cbar.set_label('Correlation Coefficient', fontsize=9)
            axes[1, 1].text(0.02, 0.98, 'Red = positive correlation, Blue = negative', 
                           transform=axes[1, 1].transAxes, fontsize=8, 
                           verticalalignment='top', style='italic', color='gray')
        else:
            axes[1, 1].text(0.5, 0.5, f'Too many dimensions ({terrain_features.shape[1]})\nSkipping correlation matrix',
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/feature_statistics.png', dpi=300, bbox_inches='tight')
        print(f"统计图已保存到: {self.save_dir}/feature_statistics.png")
        plt.close()
    
    def run_visualization(self):
        """运行完整可视化流程"""
        print("=" * 60)
        print("开始地形特征可视化...")
        print("=" * 60)
        
        # 加载模型和环境
        env, ppo_runner = self.load_model_and_env()
        if env is None or ppo_runner is None:
            return
        
        # 收集地形特征
        terrain_features = self.collect_terrain_features(env, ppo_runner)
        if terrain_features is None:
            return
        
        # t-SNE可视化
        features_2d = self.visualize_with_tsne(terrain_features)
        
        # 特征统计
        self.visualize_feature_statistics(terrain_features)
        
        # 生成分析报告
        self._generate_analysis_report(terrain_features, features_2d)
        
        print("\n" + "=" * 60)
        print("可视化完成！")
        print("=" * 60)
        print(f"结果保存在: {self.save_dir}")
        print("\n生成的文件:")
        print("  - terrain_features_tsne.png: t-SNE可视化图")
        print("  - feature_statistics.png: 特征统计图")
        print("  - analysis_report.txt: 详细分析报告")
        print("  - features_2d.npy: 降维后的2D特征")
        print("  - terrain_features.npy: 原始地形特征")
    
    def _generate_analysis_report(self, terrain_features, features_2d):
        """生成详细的分析报告"""
        print("\n正在生成分析报告...")
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("TERRAIN FEATURE VISUALIZATION ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # 1. 基本统计
        report_lines.append("1. BASIC STATISTICS")
        report_lines.append("-" * 70)
        report_lines.append(f"  Total samples: {len(terrain_features)}")
        report_lines.append(f"  Feature dimensions: {terrain_features.shape[1]}")
        report_lines.append(f"  Feature value range: [{terrain_features.min():.4f}, {terrain_features.max():.4f}]")
        report_lines.append(f"  Feature mean: {terrain_features.mean():.4f}")
        report_lines.append(f"  Feature std: {terrain_features.std():.4f}")
        report_lines.append("")
        
        # 2. 特征维度分析
        feature_means = np.mean(terrain_features, axis=0)
        feature_stds = np.std(terrain_features, axis=0)
        
        report_lines.append("2. FEATURE DIMENSION ANALYSIS")
        report_lines.append("-" * 70)
        report_lines.append("  Dimensions with highest variation (std > 1.0):")
        high_var_dims = np.where(feature_stds > 1.0)[0]
        if len(high_var_dims) > 0:
            for dim in high_var_dims:
                report_lines.append(f"    Dimension {dim}: mean={feature_means[dim]:.4f}, std={feature_stds[dim]:.4f}")
        else:
            report_lines.append("    None (all dimensions have low variation)")
        
        report_lines.append("")
        report_lines.append("  Dimensions with low variation (std < 0.1):")
        low_var_dims = np.where(feature_stds < 0.1)[0]
        if len(low_var_dims) > 0:
            for dim in low_var_dims:
                report_lines.append(f"    Dimension {dim}: mean={feature_means[dim]:.4f}, std={feature_stds[dim]:.4f}")
            report_lines.append("    WARNING: These dimensions may have collapsed!")
        else:
            report_lines.append("    None (good - all dimensions are being used)")
        report_lines.append("")
        
        # 3. t-SNE聚类分析
        report_lines.append("3. t-SNE CLUSTERING ANALYSIS")
        report_lines.append("-" * 70)
        
        # 计算点之间的平均距离（粗略估计聚类程度）
        try:
            from scipy.spatial.distance import pdist
            sample_size = min(500, len(features_2d))  # 采样以减少计算量
            sample_indices = np.random.choice(len(features_2d), sample_size, replace=False)
            sample_2d = features_2d[sample_indices]
            distances = pdist(sample_2d)
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            report_lines.append(f"  Mean distance between points: {mean_distance:.2f}")
            report_lines.append(f"  Std of distances: {std_distance:.2f}")
            report_lines.append(f"  Coefficient of variation: {std_distance/mean_distance:.3f}")
            
            # 评估聚类质量
            if std_distance/mean_distance > 0.5:
                report_lines.append("  Assessment: Points are spread out (no clear clusters)")
            else:
                report_lines.append("  Assessment: Points may form clusters")
        except ImportError:
            report_lines.append("  (scipy not available for distance calculation)")
        report_lines.append("")
        
        # 4. 总体评估
        report_lines.append("5. OVERALL ASSESSMENT")
        report_lines.append("-" * 70)
        
        # 评估指标
        issues = []
        positives = []
        
        # 检查潜在空间坍塌
        if len(low_var_dims) > terrain_features.shape[1] * 0.3:  # 超过30%的维度坍塌
            issues.append(f"Potential space collapse: {len(low_var_dims)}/{terrain_features.shape[1]} dimensions have very low variation")
        else:
            positives.append("No significant space collapse detected")
        
        # 检查特征利用
        if feature_stds.mean() < 0.5:
            issues.append("Low feature utilization: average std is too low")
        else:
            positives.append("Features show good variation")
        
        # 检查分布
        if abs(feature_means.mean()) > 0.1:
            issues.append(f"Feature distribution is biased: mean={feature_means.mean():.4f}")
        else:
            positives.append("Feature distribution is centered around zero (good for VAE)")
        
        # 输出评估
        if positives:
            report_lines.append("  POSITIVE INDICATORS:")
            for pos in positives:
                report_lines.append(f"    + {pos}")
            report_lines.append("")
        
        if issues:
            report_lines.append("  POTENTIAL ISSUES:")
            for issue in issues:
                report_lines.append(f"    - {issue}")
            report_lines.append("")
        
        # 总体结论
        report_lines.append("  CONCLUSION:")
        if len(issues) == 0:
            report_lines.append("    The encoder appears to be learning reasonable terrain representations.")
            report_lines.append("    Features show good variation and proper distribution.")
        elif len(issues) <= 1:
            report_lines.append("    The encoder is learning, but there may be room for improvement.")
            report_lines.append("    Consider adjusting beta value or training longer.")
        else:
            report_lines.append("    The encoder may not be learning effective terrain representations.")
            report_lines.append("    Recommendations:")
            report_lines.append("      - Check if beta value is too high (causing over-regularization)")
            report_lines.append("      - Check if reconstruction loss is too high")
            report_lines.append("      - Consider training for more iterations")
            report_lines.append("      - Check if the encoder architecture is appropriate")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        with open(f'{self.save_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # 打印报告
        print("\n" + report_text)
        print(f"\n详细报告已保存到: {self.save_dir}/analysis_report.txt")

def main():
    parser = argparse.ArgumentParser(
        description='地形特征t-SNE可视化工具 - 只需提供模型路径即可使用',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 最简单的方式（只需模型路径）
  python visualize_terrain_features.py --model_path logs/rough_lite3/Nov05_15-49-31_Dwaq/model_4000.pt
  
  # 指定更多样本（更精确的可视化）
  python visualize_terrain_features.py --model_path logs/rough_lite3/Nov05_15-49-31_Dwaq/model_4000.pt --num_samples 5000
        """
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径 (必需)')
    parser.add_argument('--task', type=str, default=None,
                       help='任务名称 (可选，会自动从路径检测)')
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='采样数量 (默认: 2000)\n'
                            '建议值:\n'
                            '  - 最少: 500 (快速预览，但可能不够准确)\n'
                            '  - 推荐: 2000-3000 (平衡准确性和速度)\n'
                            '  - 最佳: 5000+ (最准确，但计算时间长)\n'
                            '注意: 样本数影响t-SNE的perplexity和K-means的测试范围')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录 (可选，默认保存在模型目录下)')
    
    args = parser.parse_args()
    
    # 处理路径（支持相对路径和绝对路径）
    model_path = args.model_path
    if not os.path.isabs(model_path):
        # 相对路径，尝试从当前目录和rl_training目录查找
        current_dir_path = os.path.abspath(model_path)
        rl_training_path = os.path.join('/home/cjl/rl_training', model_path)
        
        if os.path.exists(current_dir_path):
            model_path = current_dir_path
        elif os.path.exists(rl_training_path):
            model_path = rl_training_path
        else:
            model_path = current_dir_path  # 使用当前目录的绝对路径用于错误提示
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"原始路径: {args.model_path}")
        print("\n尝试查找可能的模型文件...")
        
        # 尝试查找可能的模型文件
        if 'logs' in args.model_path:
            # 提取目录部分
            dir_part = os.path.dirname(args.model_path)
            if not os.path.isabs(dir_part):
                dir_part = os.path.join('/home/cjl/rl_training', dir_part)
            
            if os.path.exists(dir_part):
                print(f"\n在目录 {dir_part} 中找到以下文件:")
                try:
                    files = os.listdir(dir_part)
                    model_files = [f for f in files if f.endswith('.pt')]
                    if model_files:
                        for f in sorted(model_files)[:10]:  # 只显示前10个
                            print(f"  - {f}")
                        if len(model_files) > 10:
                            print(f"  ... 还有 {len(model_files) - 10} 个文件")
                    else:
                        print("  (未找到.pt文件)")
                except:
                    pass
        
        print("\n提示:")
        print("1. 请使用绝对路径，例如:")
        print("   /home/cjl/rl_training/logs/rough_lite3/Nov14_11-37-50_Dwaq/model_4000.pt")
        print("2. 或者确保在rl_training目录下运行，使用相对路径")
        return
    
    # 更新为正确的路径
    args.model_path = model_path
    
    print("\n" + "=" * 60)
    print("地形特征可视化工具")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    if args.task:
        print(f"任务名称: {args.task} (手动指定)")
    print(f"采样数量: {args.num_samples}")
    print("=" * 60 + "\n")
    
    # 创建可视化器
    try:
        visualizer = TerrainFeatureVisualizer(
            args.model_path, 
            args.task, 
            args.num_samples,
            args.save_dir
        )
        
        # 运行可视化
        visualizer.run_visualization()
        
        print("\n" + "=" * 60)
        print("完成！请查看生成的可视化图片")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n如果遇到问题，请检查:")
        print("1. 模型路径是否正确")
        print("2. 模型是否包含环境编码器")
        print("3. CUDA是否可用（如果使用GPU）")

if __name__ == "__main__":
    main()

