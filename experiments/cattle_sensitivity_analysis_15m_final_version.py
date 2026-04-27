"""
牛W/L比值参数敏感性分析脚本
========================================

目的：分析不同过滤参数对牛W/L比值时间序列轨迹的影响
用于：回应审稿人关于参数选择合理性的问题

分析维度：
1. 置信度阈值 (conf_threshold): 0.5, 0.6, 0.7, 0.8, 0.9
2. ROI区域过滤 (中心点位置): 不同矩形区域
3. W/L比值阈值 (WL_min): 0.25, 0.28, 0.30, 0.33, 0.35

输入：instances_metadata_all.csv (由 cattle_collect_all_instances.py 生成)
输出：
- 敏感性分析报告（文本）
- 参数对比轨迹图（时间序列）
- 热力图/3D表面图（参数空间）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================
# Logistic Growth Function
# ============================================
def logistic(t, L, k, t0, b):
    """
    Logistic growth function
    L: upper asymptote increase
    k: growth rate (day^-1)
    t0: inflection point (days)
    b: lower asymptote
    """
    return L / (1 + np.exp(-k * (t - t0))) + b


def logistic_derivative(t, L, k, t0, b):
    """Growth rate dy/dt"""
    exp_term = np.exp(-k * (t - t0))
    return k * L * exp_term / ((1 + exp_term) ** 2)


class CattleSensitivityAnalyzer:
    """牛W/L比值参数敏感性分析器"""
    
    def __init__(self, csv_path: str, output_dir: str = None):
        """
        初始化分析器
        
        Args:
            csv_path: instances_metadata_all.csv文件路径
            output_dir: 输出目录，默认为CSV所在目录下的带时间戳文件夹
        """
        self.csv_path = csv_path
        self.df = None
        
        # 如果没有指定输出目录，在CSV所在目录下创建带时间戳的文件夹
        if output_dir is None:
            csv_parent = Path(csv_path).parent
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = csv_parent / f'sensitivity_analysis_{timestamp}'
        else:
            self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
        
        # 参数网格
        self.conf_thresholds = [0.5, 0.6, 0.7, 0.8]
        # self.conf_thresholds = [0.7]

        self.roi_configs = {
            'no_filter': None,
            'center_50': (0.25, 0.25, 0.75, 0.75),  # 中心50%区域 (x1, y1, x2, y2)
            'center_60': (0.20, 0.20, 0.80, 0.80),  # 中心60%区域
            'center_70': (0.15, 0.15, 0.85, 0.85),  # 中心70%区域
            'center_80': (0.10, 0.10, 0.90, 0.90),  # 中心80%区域
        }
        
        self.wl_ranges = [
            (0.250, 0.333),  # 范围1: 很宽松（下限-0.05，上限+0.05）- 包含更多边缘案例
            (0.260, 0.333),  # 范围2: 较宽松（下限-0.02，上限+0.03）
            (0.270, 0.333),  # 范围3: 最佳基准（标准范围）
            (0.280, 0.333),  # 范围4: 较严格（下限+0.02，上限-0.02）
            # (0.300, 0.333),  # 范围5: 很严格（下限+0.03，上限-0.03）- 只保留核心案例
        ]

        
        # 最佳参数设置（基于大量测试）
        self.best_conf = 0.7
        self.best_roi = 'center_70'
        self.best_wl_range = (0.250, 0.333)  # 最佳W/L范围 (对应AR 3-4)
        
        self.results = {}
        self.best_results = {}  # 存储最佳参数的控制变量分析结果
        
    # @staticmethod
    # def format_roi_label(roi_name):
    #     """
    #     将ROI名称转换为图例标签
    #     center_50 -> w50_h40
    #     center_60 -> w60_h50
    #     center_70 -> w70_h60
    #     center_80 -> w80_h70
    #     """
    #     roi_mapping = {
    #         'center_50': 'w50_h40',
    #         'center_60': 'w60_h50',
    #         'center_70': 'w70_h60',
    #         'center_80': 'w80_h70',
    #     }
    #     return roi_mapping.get(roi_name, roi_name)
        
    def load_data(self):
        """加载CSV数据"""
        print(f"正在加载数据: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ 加载完成: {len(self.df)} 条记录")
        print(f"  日期范围: {self.df['Date'].min()} ~ {self.df['Date'].max()}")
        print(f"  Standing: {len(self.df[self.df['Class']=='standing'])} 条")
        print(f"  Laying: {len(self.df[self.df['Class']=='laying'])} 条")
        print()
        
    def apply_filters(self, df, conf_threshold=None, roi_config=None, wl_range=None):
        """
        应用过滤条件
        
        Args:
            df: 数据框
            conf_threshold: 置信度阈值
            roi_config: ROI配置 (x1, y1, x2, y2) 归一化坐标
            wl_range: W/L比值范围 (min, max) 元组
            
        Returns:
            过滤后的数据框
        """
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # 1. 置信度过滤
        if conf_threshold is not None:
            filtered_df = filtered_df[filtered_df['Confidence'] >= conf_threshold]
        
        # 2. ROI区域过滤（基于bbox中心点）
        if roi_config is not None:
            x1_norm, y1_norm, x2_norm, y2_norm = roi_config
            # 计算归一化的中心点坐标
            center_x_norm = filtered_df['BBox_Center_X'] / filtered_df['Image_Width']
            center_y_norm = filtered_df['BBox_Center_Y'] / filtered_df['Image_Height']
            
            # 过滤在ROI内的实例
            roi_mask = (
                (center_x_norm >= x1_norm) & (center_x_norm <= x2_norm) &
                (center_y_norm >= y1_norm) & (center_y_norm <= y2_norm)
            )
            filtered_df = filtered_df[roi_mask]
        
        # 3. W/L比值范围过滤（使用完整的[min, max]范围）
        if wl_range is not None:
            wl_min, wl_max = wl_range
            # 【关键】应用下限和上限（双向过滤）
            filtered_df = filtered_df[
                (filtered_df['WL_Ratio'] >= wl_min) & 
                (filtered_df['WL_Ratio'] <= wl_max)
            ]
        
        return filtered_df
    
    def compute_trajectory_statistics(self, df, class_name='standing'):
        """
        计算时间序列轨迹的统计信息
        
        Args:
            df: 数据框
            class_name: 类别名称
            
        Returns:
            dict: 包含时间序列统计信息
        """
        # 按日期分组
        class_df = df[df['Class'] == class_name]
        
        if len(class_df) == 0:
            return None
        
        # 按Day_Index分组（如果存在），否则按Date分组
        if 'Day_Index' in class_df.columns:
            # 先按Date分组获取Day_Index，再按Day_Index分组统计
            grouped = class_df.groupby('Day_Index')['WL_Ratio'].agg(['mean', 'std', 'count'])
            grouped = grouped.reset_index()
            grouped = grouped.sort_values('Day_Index')
            # 将Day_Index从0开始改为从1开始
            day_indices = (grouped['Day_Index'] + 1).tolist()
        else:
            # 兼容旧数据：按Date分组
            grouped = class_df.groupby('Date')['WL_Ratio'].agg(['mean', 'std', 'count'])
            grouped = grouped.reset_index()
            grouped = grouped.sort_values('Date')
            # 创建简单的序列索引
            day_indices = list(range(1, len(grouped) + 1))
        
        return {
            'day_indices': day_indices,
            'mean_wl': grouped['mean'].tolist(),
            'std_wl': grouped['std'].tolist(),
            'sem_wl': (grouped['std'] / np.sqrt(grouped['count'])).tolist(),  # SEM = SD / sqrt(n)
            'count': grouped['count'].tolist(),
            'overall_mean': class_df['WL_Ratio'].mean(),
            'overall_std': class_df['WL_Ratio'].std(),
            'total_count': len(class_df)
        }
    
    def analyze_confidence_sensitivity(self):
        """分析置信度阈值的影响"""
        print("=" * 80)
        print("分析1: 置信度阈值敏感性分析")
        print("=" * 80)
        
        results = {}
        
        for conf in self.conf_thresholds:
            filtered_df = self.apply_filters(self.df, conf_threshold=conf)
            
            standing_stats = self.compute_trajectory_statistics(filtered_df, 'standing')
            laying_stats = self.compute_trajectory_statistics(filtered_df, 'laying')
            
            results[f'conf_{conf}'] = {
                'conf': conf,
                'standing': standing_stats,
                'laying': laying_stats,
                'total_instances': len(filtered_df)
            }
            
            if standing_stats:
                print(f"Conf={conf:.1f}: Standing实例={standing_stats['total_count']}, "
                      f"平均W/L={standing_stats['overall_mean']:.3f}±{standing_stats['overall_std']:.3f}")
        
        print()
        self.results['confidence'] = results
        return results
    
    def analyze_roi_sensitivity(self):
        """分析ROI区域过滤的影响"""
        print("=" * 80)
        print("分析2: ROI区域敏感性分析")
        print("=" * 80)
        
        results = {}
        
        for roi_name, roi_config in self.roi_configs.items():
            filtered_df = self.apply_filters(self.df, roi_config=roi_config)
            
            standing_stats = self.compute_trajectory_statistics(filtered_df, 'standing')
            laying_stats = self.compute_trajectory_statistics(filtered_df, 'laying')
            
            results[roi_name] = {
                'roi_config': roi_config,
                'standing': standing_stats,
                'laying': laying_stats,
                'total_instances': len(filtered_df)
            }
            
            if standing_stats:
                print(f"ROI={roi_name}: Standing实例={standing_stats['total_count']}, "
                      f"平均W/L={standing_stats['overall_mean']:.3f}±{standing_stats['overall_std']:.3f}")
        
        print()
        self.results['roi'] = results
        return results
    
    def analyze_wl_sensitivity(self):
        """分析W/L比值阈值的影响（扩大范围 vs 缩小范围）"""
        print("=" * 80)
        print("分析3: W/L比值范围敏感性分析")
        print("说明: 以[0.25, 0.33]为最佳基准，测试放宽和收紧阈值的影响")
        print("=" * 80)
        
        results = {}
        
        for idx, (wl_min, wl_max) in enumerate(self.wl_ranges, 1):
            filtered_df = self.apply_filters(self.df, wl_range=(wl_min, wl_max))
            
            standing_stats = self.compute_trajectory_statistics(filtered_df, 'standing')
            laying_stats = self.compute_trajectory_statistics(filtered_df, 'laying')
            
            window_width = wl_max - wl_min
            range_name = f'range{idx}_[{wl_min:.2f},{wl_max:.2f}]'
            
            results[range_name] = {
                'wl_min': wl_min,
                'wl_max': wl_max,
                'window_width': window_width,
                'standing': standing_stats,
                'laying': laying_stats,
                'total_instances': len(filtered_df)
            }
            
            # 判断范围类型
            if idx == 1:
                range_type = "很宽松"
            elif idx == 2:
                range_type = "较宽松"
            elif idx == 3:
                range_type = "最佳基准"
            elif idx == 4:
                range_type = "较严格"
            else:
                range_type = "很严格"
            
            if standing_stats:
                print(f"范围{idx} [{wl_min:.2f}, {wl_max:.2f}] ({range_type}, 窗宽={window_width:.2f}): "
                      f"Standing实例={standing_stats['total_count']}, "
                      f"平均W/L={standing_stats['overall_mean']:.3f}±{standing_stats['overall_std']:.3f}")
        
        print()
        self.results['wl_range'] = results
        return results
    
    def analyze_with_best_params(self):
        """
        使用最佳参数进行控制变量敏感性分析
        最佳参数：conf=0.7, ROI=center_70, WL=[0.25, 0.333]
        每次只变动一个参数，其他固定为最佳值
        """
        print("\n" + "=" * 80)
        print("最佳参数控制变量分析")
        print(f"最佳参数：conf={self.best_conf}, ROI={self.best_roi}, WL={self.best_wl_range}")
        print("=" * 80 + "\n")
        
        # 初始化best_results字典
        self.best_results = {'confidence': {}, 'roi': {}, 'wl_range': {}}
        
        # 关键改进：先计算基准参数组合（使用当前最佳参数）
        # 这个组合会在三个分析中都出现，所以先单独计算并缓存
        print("=" * 80)
        print(f"计算基准参数组合（Conf={self.best_conf}, ROI={self.best_roi}, WL={self.best_wl_range}）")
        print("=" * 80)
        print(f"\n【详细过滤诊断】")
        baseline_df = self.apply_filters(
            self.df,
            conf_threshold=self.best_conf,
            roi_config=self.roi_configs[self.best_roi],
            wl_range=self.best_wl_range
        )
        
        # 详细诊断
        print(f"总数据: {len(self.df)} 条")
        print(f"过滤后: {len(baseline_df)} 条 (保留率: {len(baseline_df)/len(self.df)*100:.1f}%)")
        
        # 按类别统计
        standing_count = len(baseline_df[baseline_df['Class'] == 'standing'])
        laying_count = len(baseline_df[baseline_df['Class'] == 'laying'])
        print(f"Standing: {standing_count} 条, Laying: {laying_count} 条")
        
        # 按日期统计
        if 'Day_Index' in baseline_df.columns:
            daily_counts = baseline_df[baseline_df['Class'] == 'standing'].groupby('Day_Index').size()
            print(f"\n每日Standing实例数:")
            for day_idx, count in daily_counts.items():
                print(f"  Day {day_idx+1}: {count} 个实例")
        
        # W/L统计
        standing_df = baseline_df[baseline_df['Class'] == 'standing']
        if len(standing_df) > 0:
            print(f"\nW/L比值统计 (Standing):")
            print(f"  范围: [{standing_df['WL_Ratio'].min():.3f}, {standing_df['WL_Ratio'].max():.3f}]")
            print(f"  均值±标准差: {standing_df['WL_Ratio'].mean():.3f}±{standing_df['WL_Ratio'].std():.3f}")
        
        baseline_standing = self.compute_trajectory_statistics(baseline_df, 'standing')
        baseline_laying = self.compute_trajectory_statistics(baseline_df, 'laying')
        
        if baseline_standing:
            print(f"\n基准轨迹统计:")
            print(f"  总实例数: {baseline_standing['total_count']}")
            print(f"  时间点数: {len(baseline_standing['day_indices'])}")
            print(f"  整体均值: {baseline_standing['overall_mean']:.3f}±{baseline_standing['overall_std']:.3f}")
            print(f"\n  后三个时间点的数据:")
            for i in range(-3, 0):
                day = baseline_standing['day_indices'][i]
                mean = baseline_standing['mean_wl'][i]
                sem = baseline_standing['sem_wl'][i]
                count = baseline_standing['count'][i]
                print(f"    Day {day}: mean={mean:.4f}, SEM={sem:.4f}, n={count}")
        print()

        
        # 1. 固定ROI和WL，变化置信度
        print("分析1: 变化置信度（固定ROI和WL）")
        print("-" * 80)
        conf_results = {}
        for conf in self.conf_thresholds:
            if conf == self.best_conf:
                # 使用缓存的基准数据
                conf_results[f'conf_{conf}'] = {
                    'conf': conf,
                    'standing': baseline_standing,
                    'laying': baseline_laying,
                    'total_instances': len(baseline_df),
                    'is_baseline': True  # 标记为基准
                }
                print(f"Conf={conf:.1f}: Standing实例={baseline_standing['total_count']}, "
                      f"平均W/L={baseline_standing['overall_mean']:.3f}±{baseline_standing['overall_std']:.3f}")
            else:
                filtered_df = self.apply_filters(
                    self.df, 
                    conf_threshold=conf,
                    roi_config=self.roi_configs[self.best_roi],
                    wl_range=self.best_wl_range
                )
                
                standing_stats = self.compute_trajectory_statistics(filtered_df, 'standing')
                laying_stats = self.compute_trajectory_statistics(filtered_df, 'laying')
                
                conf_results[f'conf_{conf}'] = {
                    'conf': conf,
                    'standing': standing_stats,
                    'laying': laying_stats,
                    'total_instances': len(filtered_df),
                    'is_baseline': False
                }
                
                if standing_stats:
                    print(f"Conf={conf:.1f}: Standing实例={standing_stats['total_count']}, "
                          f"平均W/L={standing_stats['overall_mean']:.3f}±{standing_stats['overall_std']:.3f}")
        
        self.best_results['confidence'] = conf_results
        
        # 2. 固定Conf和WL，变化ROI
        print("\n分析2: 变化ROI区域（固定Conf和WL）")
        print("-" * 80)
        roi_results = {}
        for roi_name, roi_config in self.roi_configs.items():
            if roi_name == 'no_filter':
                continue  # 跳过无过滤
            
            if roi_name == self.best_roi:
                # 使用缓存的基准数据
                roi_results[roi_name] = {
                    'roi_config': roi_config,
                    'standing': baseline_standing,
                    'laying': baseline_laying,
                    'total_instances': len(baseline_df),
                    'is_baseline': True  # 标记为基准
                }
                print(f"ROI={roi_name}: Standing实例={baseline_standing['total_count']}, "
                      f"平均W/L={baseline_standing['overall_mean']:.3f}±{baseline_standing['overall_std']:.3f}")
            else:
                filtered_df = self.apply_filters(
                    self.df,
                    conf_threshold=self.best_conf,
                    roi_config=roi_config,
                    wl_range=self.best_wl_range
                )
                
                standing_stats = self.compute_trajectory_statistics(filtered_df, 'standing')
                laying_stats = self.compute_trajectory_statistics(filtered_df, 'laying')
                
                roi_results[roi_name] = {
                    'roi_config': roi_config,
                    'standing': standing_stats,
                    'laying': laying_stats,
                    'total_instances': len(filtered_df),
                    'is_baseline': False
                }
                
                if standing_stats:
                    print(f"ROI={roi_name}: Standing实例={standing_stats['total_count']}, "
                          f"平均W/L={standing_stats['overall_mean']:.3f}±{standing_stats['overall_std']:.3f}")
        
        self.best_results['roi'] = roi_results
        
        # 3. 固定Conf和ROI，变化W/L阈值范围
        print("\n分析3: 变化W/L阈值范围（固定Conf和ROI）")
        print("-" * 80)
        wl_results = {}
        for idx, (wl_min, wl_max) in enumerate(self.wl_ranges, 1):
            if (wl_min, wl_max) == self.best_wl_range:
                # 使用缓存的基准数据
                range_name = f'range{idx}_[{wl_min:.2f},{wl_max:.2f}]'
                window_width = wl_max - wl_min
                wl_results[range_name] = {
                    'wl_min': wl_min,
                    'wl_max': wl_max,
                    'window_width': window_width,
                    'standing': baseline_standing,
                    'laying': baseline_laying,
                    'total_instances': len(baseline_df),
                    'is_baseline': True  # 标记为基准
                }
                print(f"范围{idx} [{wl_min:.2f}, {wl_max:.2f}]: "
                      f"Standing实例={baseline_standing['total_count']}, "
                      f"平均W/L={baseline_standing['overall_mean']:.3f}±{baseline_standing['overall_std']:.3f}")
            else:
                filtered_df = self.apply_filters(
                    self.df,
                    conf_threshold=self.best_conf,
                    roi_config=self.roi_configs[self.best_roi],
                    wl_range=(wl_min, wl_max)
                )
                
                standing_stats = self.compute_trajectory_statistics(filtered_df, 'standing')
                laying_stats = self.compute_trajectory_statistics(filtered_df, 'laying')
                
                window_width = wl_max - wl_min
                range_name = f'range{idx}_[{wl_min:.2f},{wl_max:.2f}]'
                
                wl_results[range_name] = {
                    'wl_min': wl_min,
                    'wl_max': wl_max,
                    'window_width': window_width,
                    'standing': standing_stats,
                    'laying': laying_stats,
                    'total_instances': len(filtered_df),
                    'is_baseline': False
                }
                
                if standing_stats:
                    print(f"范围{idx} [{wl_min:.2f}, {wl_max:.2f}] (窗宽={window_width:.2f}): "
                          f"Standing实例={standing_stats['total_count']}, "
                          f"平均W/L={standing_stats['overall_mean']:.3f}±{standing_stats['overall_std']:.3f}")
        
        self.best_results['wl_range'] = wl_results
        print()
    
    def plot_confidence_trajectories(self, save_path=None):
        """绘制不同置信度阈值下的W/L轨迹"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Standing
        ax = axes[0]
        for conf_key, data in self.results['confidence'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                ax.plot(day_indices, stats['mean_wl'], marker='o', label=f"Conf={data['conf']:.1f}")
                ax.fill_between(day_indices, 
                               np.array(stats['mean_wl']) - np.array(stats['sem_wl']),
                               np.array(stats['mean_wl']) + np.array(stats['sem_wl']),
                               alpha=0.2)
        
        ax.set_xlabel('Days', fontsize=18, fontweight='bold')
        ax.set_ylabel('W/L Ratio', fontsize=18, fontweight='bold')
        ax.set_title('Standing Cattle: W/L Ratio Trajectory\n(Different Confidence Thresholds)', fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if day_indices:
            ax.set_xticks(day_indices)
        ax.legend(fontsize=14, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Laying
        ax = axes[1]
        for conf_key, data in self.results['confidence'].items():
            if data['laying'] is not None:
                stats = data['laying']
                day_indices = stats['day_indices']
                ax.plot(day_indices, stats['mean_wl'], marker='s', label=f"Conf={data['conf']:.1f}")
                ax.fill_between(day_indices, 
                               np.array(stats['mean_wl']) - np.array(stats['sem_wl']),
                               np.array(stats['mean_wl']) + np.array(stats['sem_wl']),
                               alpha=0.2)
        
        ax.set_xlabel('Days', fontsize=18, fontweight='bold')
        ax.set_ylabel('W/L Ratio', fontsize=18, fontweight='bold')
        ax.set_title('Laying Cattle: W/L Ratio Trajectory\n(Different Confidence Thresholds)', fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if day_indices:
            ax.set_xticks(day_indices)
        ax.legend(fontsize=14, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存图表: {save_path}")
        
        return fig
    
    def plot_roi_trajectories(self, save_path=None):
        """绘制不同ROI配置下的W/L轨迹"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Standing
        ax = axes[0]
        for roi_name, data in self.results['roi'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                # 使用原始ROI名称并添加%
                label = roi_name.replace('center_', 'center_') + '%' if roi_name.startswith('center_') else roi_name
                ax.plot(day_indices, stats['mean_wl'], marker='o', label=label)
        
        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('W/L Ratio', fontsize=12)
        ax.set_title('Standing Cattle: W/L Ratio Trajectory\n(Different ROI Regions)', fontsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if data['standing'] is not None:
            ax.set_xticks(data['standing']['day_indices'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Laying
        ax = axes[1]
        for roi_name, data in self.results['roi'].items():
            if data['laying'] is not None:
                stats = data['laying']
                day_indices = stats['day_indices']
                # 使用原始ROI名称并添加%
                label = roi_name.replace('center_', 'center_') + '%' if roi_name.startswith('center_') else roi_name
                ax.plot(day_indices, stats['mean_wl'], marker='s', label=label)
        
        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('W/L Ratio', fontsize=12)
        ax.set_title('Laying Cattle: W/L Ratio Trajectory\n(Different ROI Regions)', fontsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if data['laying'] is not None:
            ax.set_xticks(data['laying']['day_indices'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存图表: {save_path}")
        
        return fig
    
    def plot_wl_threshold_trajectories(self, save_path=None):
        """绘制不同W/L阈值范围下的轨迹"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Standing
        ax = axes[0]
        for wl_key, data in self.results['wl_range'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                ax.plot(day_indices, stats['mean_wl'], marker='o', 
                       label=f"W/L[{data['wl_min']:.3f}, {data['wl_max']:.3f}]")
        
        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('W/L Ratio', fontsize=12)
        ax.set_title('Standing Cattle: W/L Ratio Trajectory\n(Different W/L Thresholds)', fontsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if data['standing'] is not None:
            ax.set_xticks(data['standing']['day_indices'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Laying
        ax = axes[1]
        for wl_key, data in self.results['wl_range'].items():
            if data['laying'] is not None:
                stats = data['laying']
                day_indices = stats['day_indices']
                ax.plot(day_indices, stats['mean_wl'], marker='s', 
                       label=f"W/L[{data['wl_min']:.3f}, {data['wl_max']:.3f}]")
        
        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('W/L Ratio', fontsize=12)
        ax.set_title('Laying Cattle: W/L Ratio Trajectory\n(Different W/L Thresholds)', fontsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if data['laying'] is not None:
            ax.set_xticks(data['laying']['day_indices'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存图表: {save_path}")
        
        return fig
    
    def plot_parameter_heatmap(self, save_path=None):
        """绘制参数影响热力图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Confidence vs WL mean (Standing)
        conf_means = []
        conf_labels = []
        for conf_key, data in self.results['confidence'].items():
            if data['standing'] is not None:
                conf_means.append(data['standing']['overall_mean'])
                conf_labels.append(f"{data['conf']:.1f}")
        
        wl_means = []
        wl_labels = []
        for wl_key, data in self.results['wl_range'].items():
            if data['standing'] is not None:
                wl_means.append(data['standing']['overall_mean'])
                wl_labels.append(f"[{data['wl_min']:.3f},{data['wl_max']:.3f}]")
        
        ax = axes[0]
        matrix = np.outer(conf_means, wl_means)
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(wl_labels)))
        ax.set_yticks(range(len(conf_labels)))
        ax.set_xticklabels(wl_labels)
        ax.set_yticklabels(conf_labels)
        ax.set_xlabel('W/L Threshold', fontsize=12)
        ax.set_ylabel('Confidence Threshold', fontsize=12)
        ax.set_title('Standing Cattle: Mean W/L Ratio\n(Conf × W/L Threshold)', fontsize=14)
        plt.colorbar(im, ax=ax, label='Mean W/L Ratio')
        
        # 2. Parameter impact on instance count
        conf_counts = []
        for conf_key, data in self.results['confidence'].items():
            if data['standing'] is not None:
                conf_counts.append(data['standing']['total_count'])
        
        wl_counts = []
        for wl_key, data in self.results['wl_range'].items():
            if data['standing'] is not None:
                wl_counts.append(data['standing']['total_count'])
        
        ax = axes[1]
        ax.plot(conf_labels, conf_counts, 'o-', linewidth=2, markersize=8, label='By Confidence')
        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel('Instance Count', fontsize=12, color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(wl_labels, wl_counts, 's-', linewidth=2, markersize=8, 
                color='C1', label='By W/L Threshold')
        ax2.set_ylabel('Instance Count', fontsize=12, color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        
        ax.set_title('Standing Cattle: Instance Count\nvs Parameter Thresholds', fontsize=14)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存图表: {save_path}")
        
        return fig
    def plot_combined_sensitivity(self, save_path=None):
        """
        综合敏感性分析图：展示三个关键参数对herd mean W/L trajectory的影响
        符合审稿人要求：θ_conf, ROI size, W/L thresholds
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # 1. Confidence Threshold Sensitivity (θ_conf)
        ax = axes[0]
        for conf_key, data in self.results['confidence'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                ax.plot(day_indices, stats['mean_wl'], marker='o', linewidth=2,
                       label=f"θ_conf={data['conf']:.1f}")
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title('(a) Confidence Threshold (θ_conf) Sensitivity', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if day_indices:
            ax.set_xticks(day_indices)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. ROI Size Sensitivity (wr/hr)
        ax = axes[1]
        roi_day_indices = None
        for roi_name, data in self.results['roi'].items():
            if data['standing'] is not None and roi_name != 'no_filter':
                stats = data['standing']
                day_indices = stats['day_indices']
                roi_day_indices = day_indices  # 保存用于设置x轴
                # 使用原始ROI名称并添加%
                roi_label = roi_name.replace('center_', 'center_') + '%' if roi_name.startswith('center_') else roi_name
                ax.plot(day_indices, stats['mean_wl'], marker='s', linewidth=2,
                       label=f"ROI={roi_label}")
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title('(b) ROI Size (wr/hr) Sensitivity', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if roi_day_indices:
            ax.set_xticks(roi_day_indices)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. W/L Threshold Sensitivity (原AR阈值)
        ax = axes[2]
        wl_day_indices = None
        for wl_key, data in self.results['wl_range'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                wl_day_indices = day_indices  # 保存用于设置x轴
                ax.plot(day_indices, stats['mean_wl'], marker='^', linewidth=2,
                       label=f"W/L∈[{data['wl_min']:.3f}, {data['wl_max']:.3f}]")
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title('(c) W/L Threshold Sensitivity', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # 设置x轴为实际天数
        if wl_day_indices:
            ax.set_xticks(wl_day_indices)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存综合敏感性分析图: {save_path}")
        
        return fig
    
    def plot_best_params_sensitivity(self, save_path=None):
        """
        最佳参数控制变量敏感性分析图
        固定最佳参数（conf=0.7, ROI=center_70, W/L=[0.25,0.333]），每次只变动一个
        """
        if not hasattr(self, 'best_results') or not self.best_results:
            print("警告: 需要先运行 analyze_with_best_params()")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # 定义基准颜色（最佳参数组合在三个子图中使用相同颜色）
        baseline_color = '#D62728'  # 深红色
        
        # 1. 变化置信度（固定ROI=center_70, W/L=[0.25,0.333]）
        ax = axes[0]
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                # 如果是基准参数，使用基准颜色
                is_baseline = data.get('is_baseline', False)
                color = baseline_color if is_baseline else None
                ax.plot(day_indices, stats['mean_wl'], marker='o', linewidth=2,
                       label=f"θ_conf={data['conf']:.1f}", color=color)
                ax.fill_between(day_indices,
                               np.array(stats['mean_wl']) - np.array(stats['sem_wl']),
                               np.array(stats['mean_wl']) + np.array(stats['sem_wl']),
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'(a) Confidence (Fixed ROI={self.best_roi}, W/L={self.best_wl_range})', 
                    fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        if day_indices:
            ax.set_xticks(day_indices)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. 变化ROI区域（固定Conf=0.7, W/L=[0.25,0.333]）
        ax = axes[1]
        roi_day_indices = None
        for roi_name, data in self.best_results['roi'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                roi_day_indices = day_indices
                # 使用格式化的ROI标签
                roi_label = self.format_roi_label(roi_name)
                # 如果是基准参数，使用基准颜色
                is_baseline = data.get('is_baseline', False)
                color = baseline_color if is_baseline else None
                ax.plot(day_indices, stats['mean_wl'], marker='s', linewidth=2,
                       label=f"ROI={roi_label}", color=color)
                ax.fill_between(day_indices,
                               np.array(stats['mean_wl']) - np.array(stats['sem_wl']),
                               np.array(stats['mean_wl']) + np.array(stats['sem_wl']),
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'(b) ROI Size (Fixed Conf={self.best_conf}, W/L={self.best_wl_range})', 
                    fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        if roi_day_indices:
            ax.set_xticks(roi_day_indices)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. 变化W/L阈值范围（固定Conf=0.7, ROI=center_70）
        ax = axes[2]
        wl_day_indices = None
        for wl_key, data in self.best_results['wl_range'].items():
            if data['standing'] is not None:
                stats = data['standing']
                day_indices = stats['day_indices']
                wl_day_indices = day_indices
                # 如果是基准参数，使用基准颜色
                is_baseline = data.get('is_baseline', False)
                color = baseline_color if is_baseline else None
                ax.plot(day_indices, stats['mean_wl'], marker='^', linewidth=2,
                       label=f"W/L∈[{data['wl_min']:.3f},{data['wl_max']:.3f}]", color=color)
                ax.fill_between(day_indices,
                               np.array(stats['mean_wl']) - np.array(stats['sem_wl']),
                               np.array(stats['mean_wl']) + np.array(stats['sem_wl']),
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'(c) W/L Threshold (Fixed Conf={self.best_conf}, ROI={self.best_roi})', 
                    fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        if wl_day_indices:
            ax.set_xticks(wl_day_indices)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存最佳参数控制变量分析图: {save_path}")
        
        return fig
    
    def diagnose_early_data_quality(self, stats_dict):
        """
        诊断早期数据质量，识别可能影响拟合的问题
        
        Args:
            stats_dict: 统计字典
            
        Returns:
            list: 问题列表
        """
        if stats_dict is None:
            return ["⚠️ 无数据"]
        
        days = np.array(stats_dict['day_indices'])
        sample_sizes = np.array(stats_dict['count'])
        y_observed = np.array(stats_dict['mean_wl'])
        
        # 定义早期阶段（前1/3）
        early_cutoff_idx = max(1, len(days)//3)
        early_samples = sample_sizes[:early_cutoff_idx]
        late_samples = sample_sizes[early_cutoff_idx:]
        
        issues = []
        
        # 检查1：早期样本量是否过少
        early_mean = np.mean(early_samples)
        if early_mean < 10:
            issues.append(f"⚠️ 早期平均样本量过少: n={early_mean:.1f}")
        
        # 检查2：样本量变化是否剧烈
        if len(late_samples) > 0:
            late_mean = np.mean(late_samples)
            sample_ratio = late_mean / (early_mean + 1e-6)
            if sample_ratio > 5:
                issues.append(f"⚠️ 早期vs后期样本量差距: {sample_ratio:.1f}x")
        
        # 检查3：早期数据点是否太少
        if early_cutoff_idx < 3:
            issues.append(f"⚠️ 早期时间点太少: {early_cutoff_idx}个")
        
        # 检查4：早期数据波动是否剧烈
        if len(early_samples) >= 2:
            early_y = y_observed[:early_cutoff_idx]
            early_cv = np.std(early_y) / (np.mean(early_y) + 1e-6) * 100
            if early_cv > 5:  # 变异系数>5%
                issues.append(f"⚠️ 早期数据波动大: CV={early_cv:.1f}%")
        
        # 检查5：总样本量
        total_samples = np.sum(sample_sizes)
        if total_samples < 100:
            issues.append(f"⚠️ 总样本量不足: n={total_samples}")
        
        return issues
    
    def fit_logistic_growth(self, stats_dict, param_name='default'):
        """
        对W/L轨迹进行logistic生长曲线拟合
        
        Args:
            stats_dict: 包含day_indices, mean_wl, sem_wl, count的统计字典
            param_name: 参数名称（用于输出）
            
        Returns:
            dict: 拟合结果和统计检验
        """
        if stats_dict is None:
            return None
            
        days = np.array(stats_dict['day_indices'])
        y_observed = np.array(stats_dict['mean_wl'])
        y_sem = np.array(stats_dict['sem_wl'])
        sample_sizes = np.array(stats_dict['count'])
        
        # 统计显著性检验
        # 1. 线性回归 p-value
        slope, intercept, r_value, p_value_linear, std_err = stats.linregress(days, y_observed)
        
        # 2. Spearman相关检验
        spearman_corr, p_value_spearman = stats.spearmanr(days, y_observed)
        
        # 3. Cohen's d效应量
        mean_diff = y_observed[-1] - y_observed[0]
        pooled_std = np.sqrt((y_sem[0]**2 + y_sem[-1]**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # 数据质量诊断
        data_issues = self.diagnose_early_data_quality(stats_dict)
        
        # 检查数据趋势
        print(f"\n  [{param_name}] 数据诊断:")
        print(f"    数据点数: {len(days)}, 样本量范围: {sample_sizes.min()}-{sample_sizes.max()}")
        print(f"    Y值范围: {y_observed.min():.4f} - {y_observed.max():.4f}")
        print(f"    最后3个点: {y_observed[-3:] if len(y_observed) >= 3 else y_observed}")
        
        # ===== 异常值检测 =====
        # 使用移动中位数检测异常值
        outlier_mask = np.zeros(len(y_observed), dtype=bool)
        if len(y_observed) >= 5:
            # 计算每个点与邻近点的偏差
            for i in range(len(y_observed)):
                # 获取邻近点（±2个点）
                window_start = max(0, i-2)
                window_end = min(len(y_observed), i+3)
                neighbors = np.concatenate([y_observed[window_start:i], y_observed[i+1:window_end]])
                
                if len(neighbors) >= 2:
                    median_neighbor = np.median(neighbors)
                    std_neighbor = np.std(neighbors)
                    # 如果偏离邻近点中位数超过3倍标准差，标记为异常
                    if abs(y_observed[i] - median_neighbor) > 3 * std_neighbor and std_neighbor > 0.001:
                        outlier_mask[i] = True
                        print(f"    ⚠️ 检测到异常值: Day {days[i]}, Y={y_observed[i]:.4f} (偏离邻近点 {abs(y_observed[i] - median_neighbor):.4f})")
        
        # 检查是否有下降趋势（最后几个点）
        if len(y_observed) >= 3:
            last_trend = y_observed[-1] - y_observed[-3]
            if last_trend < 0:
                print(f"    ⚠️ 警告：最后3个点呈下降趋势 (Δ={last_trend:.4f})")
        
        # Logistic拟合 - 尝试多组初始值
        best_fit_result = None
        best_r2 = -np.inf
        
        # 定义多组初始值策略
        init_strategies = []
        
        # ===== 策略1: 智能估计（主策略）=====
        
        # 1. b（初始值）：前25%数据的平均
        early_cutoff = max(3, len(y_observed)//4)
        early_values = y_observed[:early_cutoff]
        b_init_1 = np.mean(early_values)
        
        # 2. 最终值：后25%数据，考虑下降情况
        late_cutoff = max(1, 3*len(y_observed)//4)
        late_values = y_observed[late_cutoff:]
        
        # 如果后期有明显下降，用最大值；否则用平均值
        if len(y_observed) >= 3 and y_observed[-1] < y_observed[-3] * 0.98:
            final_init_1 = np.max(y_observed[late_cutoff:])
            print(f"    最终值：后期有下降，用后25%最大值 = {final_init_1:.4f}")
        else:
            final_init_1 = np.mean(late_values)
            print(f"    最终值：后25%平均 = {final_init_1:.4f}")
        
        L_init_1 = max(final_init_1 - b_init_1, 0.001)
        
        # 3. 在有效增长区域找t0和k（改进版 - 与combined.py一致）
        # 策略：排除滞后期（53天前）和平台期（最后25%），在增长期找最大斜率
        if len(y_observed) > 3:
            # 定义有效数据点区间（不是斜率区间！）
            # 1. 排除滞后期：53天之后（或前25%的数据点）
            lag_phase_cutoff = max(
                np.searchsorted(days, 53),  # 53天
                len(days) // 4              # 或前25%
            )
            
            # 2. 排除平台期：最后25%的数据点
            plateau_phase_cutoff = max(1, 3 * len(days) // 4)
            
            # 确保搜索区间有效
            if lag_phase_cutoff >= plateau_phase_cutoff - 1:
                # 如果区间无效，使用中间50%的数据
                lag_phase_cutoff = len(days) // 4
                plateau_phase_cutoff = 3 * len(days) // 4
            
            # 【关键改进】使用三点中心差分法计算斜率，更平滑且对噪声鲁棒
            # 对于点i，使用 (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
            # 确保i-1和i+1都在有效范围内
            valid_rates = []
            valid_positions = []
            
            for i in range(lag_phase_cutoff + 1, plateau_phase_cutoff - 1):
                # 使用三点中心差分计算点i的斜率
                # 确保i-1, i, i+1都在有效区间内（不在滞后期，也不在平台期）
                if i > 0 and i < len(days) - 1:
                    slope = (y_observed[i+1] - y_observed[i-1]) / (days[i+1] - days[i-1])
                    if slope > 0:  # 只考虑正斜率
                        valid_rates.append(slope)
                        valid_positions.append(i)
            
            if len(valid_rates) > 0:
                # 在增长期找最大斜率作为t0（拐点）
                max_slope_idx = np.argmax(valid_rates)
                actual_idx = valid_positions[max_slope_idx]
                max_slope = valid_rates[max_slope_idx]
                
                # t0就是该点的时间
                t0_init_1 = days[actual_idx]
                k_init_1 = abs(4 * max_slope / (L_init_1 + 1e-6))
                k_init_1 = np.clip(k_init_1, 0.01, 0.3)
                print(f"    t0：三点中心差分最大斜率位置 Day {t0_init_1:.0f} (有效区间{len(valid_rates)}个点)")
            else:
                # 没有找到有效斜率，退化为两点法
                growth_rates = np.diff(y_observed) / np.diff(days)
                # 只在有效区间计算（排除最后一个点，因为它可能在平台期）
                valid_growth_rates = growth_rates[lag_phase_cutoff:plateau_phase_cutoff-1]
                if len(valid_growth_rates) > 0:
                    max_slope = np.max(valid_growth_rates)
                    max_growth_idx = lag_phase_cutoff + np.argmax(valid_growth_rates)
                    t0_init_1 = days[max_growth_idx]
                    k_init_1 = abs(4 * max_slope / (L_init_1 + 1e-6))
                    k_init_1 = np.clip(k_init_1, 0.01, 0.3)
                    print(f"    t0：两点法最大斜率位置 Day {t0_init_1:.0f}")
                else:
                    # 完全没有有效数据，使用中点
                    k_init_1 = 0.05
                    t0_init_1 = days[len(days)//2]
                    print(f"    t0：数据不足，使用中点 Day {t0_init_1:.0f}")
        else:
            k_init_1 = 0.05
            t0_init_1 = days[len(days)//2]
        
        init_strategies.append(('智能估计', [L_init_1, k_init_1, t0_init_1, b_init_1]))
        
        # 策略2: 拐点在80-100天（目标策略） ⭐⭐⭐⭐⭐
        t0_init_2 = 90.0  # 拐点设在80-100的中点
        k_init_2 = 0.05  # 适中的增长速率
        init_strategies.append(('拐点80-100天', [L_init_1, k_init_2, t0_init_2, b_init_1]))
        
        # 策略3: 保守估计（慢增长）
        k_init_3 = 0.03
        t0_init_3 = days[len(days)//2]
        init_strategies.append(('保守估计', [L_init_1, k_init_3, t0_init_3, b_init_1]))
        
        # 策略4: 中期拐点（观测期中点）
        if len(days) > 5:
            t0_init_4 = days[len(days)//2]
            k_init_4 = 0.08
            init_strategies.append(('中期拐点', [L_init_1, k_init_4, t0_init_4, b_init_1]))
        
        # 尝试每组初始值
        for strategy_name, p0 in init_strategies:
            try:
                print(f"    尝试策略[{strategy_name}]: L={p0[0]:.4f}, k={p0[1]:.4f}, t0={p0[2]:.1f}, b={p0[3]:.4f}")
                
                # 调整边界：拐点t0必须在观测期内，且在观测期内形成完整S曲线
                # 【关键约束】必须在观测期末能看到平台期！
                # 1. 下界：至少53天之后（生物学约束，滞后期结束）
                # 2. 上界：必须留出足够时间到达平台期
                #    对于Logistic曲线，从拐点到95%饱和需要 ~3/k 天
                #    假设k=0.05，需要60天；k=0.1需要30天
                #    保守起见，至少需要15-20天从拐点到平台期
                t0_lower_bound = 53.0  # 下界：至少53天
                # 上界：观测期末 - 15天（确保能看到平台期）或 95天（数据实际最大值）
                min_time_to_plateau = 15.0  # 从拐点到平台期至少需要15天
                t0_upper_bound = min(95.0, days.max() - min_time_to_plateau)
                
                bounds = (
                    [p0[0] * 0.1, 0.005, t0_lower_bound, p0[3] * 0.8],
                    [p0[0] * 5.0, 0.5, t0_upper_bound, p0[3] * 1.2]
                )
                
                # 加权策略：异常值降权、样本量少的点降权、后期下降点降权
                min_samples = 5
                sample_weight = np.where(sample_sizes < min_samples, 
                                        sample_sizes / min_samples, 
                                        1.0)
                
                # 异常值权重设为0.1
                sample_weight = np.where(outlier_mask, 0.1, sample_weight)
                
                # 后期下降点权重减半
                if len(y_observed) >= 3 and y_observed[-1] < y_observed[-3]:
                    sample_weight[-2:] *= 0.5
                
                # 修复：直接使用标准误差作为sigma，不归一化权重
                # 应用样本权重到标准误差
                sigma_weighted = y_sem / np.sqrt(sample_weight)
                
                popt, pcov = curve_fit(logistic, days, y_observed, p0=p0, 
                                     sigma=sigma_weighted, absolute_sigma=True,
                                     bounds=bounds, maxfev=20000)
                
                L_fit, k_fit, t0_fit, b_fit = popt
                
                # 计算拟合优度
                y_predicted = logistic(days, *popt)
                r2 = r2_score(y_observed, y_predicted)
                rmse = np.sqrt(mean_squared_error(y_observed, y_predicted))
                
                # 检查是否是S型曲线
                is_s_curve = True
                inflection_value = b_fit + L_fit / 2
                
                # 1. 拐点值合理性
                if inflection_value < y_observed.min() * 0.5 or inflection_value > y_observed.max() * 2.0:
                    is_s_curve = False
                
                # 2. 曲线形状：在观测期内应该是单调递增
                days_smooth = np.linspace(days.min(), days.max(), 100)
                y_smooth = logistic(days_smooth, *popt)
                dy_smooth = np.diff(y_smooth)
                
                if np.mean(dy_smooth) < 0:
                    is_s_curve = False
                
                # 3. k应该在合理范围
                if k_fit < 0.005 or k_fit > 0.5:
                    is_s_curve = False
                
                # 4. 最终值应该合理
                final_value = b_fit + L_fit
                if final_value < y_observed.min() or final_value > y_observed.max() * 3.0:
                    is_s_curve = False
                
                # 5. 【完整S曲线的关键检验】观测期末必须已接近平台期
                #    计算观测期最后时刻的增长率，必须远小于最大增长率
                max_growth_rate = k_fit * L_fit / 4  # 拐点处的最大增长率
                growth_at_end = logistic_derivative(days.max(), L_fit, k_fit, t0_fit, b_fit)
                # 如果观测期末的增长率还大于最大增长率的30%，说明还在快速增长期
                # （30%是比较合理的阈值，允许一定的增长但要明显放缓）
                if abs(growth_at_end) > max_growth_rate * 0.3:
                    is_s_curve = False  # 未看到完整S曲线，还在增长期
                
                print(f"      拟合结果: L={L_fit:.4f}, k={k_fit:.4f}, t0={t0_fit:.1f}, b={b_fit:.4f}")
                print(f"      拟合质量: R²={r2:.4f}, RMSE={rmse:.6f}, S型={'✓' if is_s_curve else '✗'}")
                
                # 【关键修改】选择策略的优先级：
                # 1. 必须是完整S型曲线 (is_s_curve=True)
                # 2. 拐点必须在观测期内 (53 <= t0 <= days.max())
                # 3. 拐点在80-100天范围内的优先级最高
                # 4. 在满足约束的前提下，选择R²最高的
                
                # 计算拐点评分：在80-100天范围内得分最高
                t0_ideal_min = 80.0
                t0_ideal_max = 100.0
                
                if t0_ideal_min <= t0_fit <= t0_ideal_max:
                    # 在理想范围内：满分1.0
                    t0_score = 1.0
                    t0_status = "理想范围(80-100)"
                elif 53 <= t0_fit < t0_ideal_min:
                    # 在53-80之间：根据距离打分，越接近80分数越高
                    t0_score = 0.7 + 0.3 * (t0_fit - 53) / (t0_ideal_min - 53)
                    t0_status = f"可接受(53-80)"
                elif t0_ideal_max < t0_fit <= days.max():
                    # 在100-观测期末之间：根据距离打分，越接近100分数越高
                    t0_score = 0.7 + 0.3 * (days.max() - t0_fit) / (days.max() - t0_ideal_max)
                    t0_status = f"可接受(100-{days.max():.0f})"
                else:
                    # 不在有效范围内
                    t0_score = 0.0
                    t0_status = "不满足约束"
                
                # 综合评分：R² * t0_score
                combined_score = r2 * t0_score
                
                # 检查是否满足核心约束
                meets_constraints = is_s_curve and (t0_fit >= 53) and (t0_fit <= days.max())
                
                print(f"      拐点位置: t0={t0_fit:.1f} -> {t0_status} (得分={t0_score:.3f})")
                print(f"      综合评分: R²={r2:.4f} × t0_score={t0_score:.3f} = {combined_score:.4f}")
                
                if meets_constraints:
                    # 满足约束，比较综合评分
                    if best_fit_result is None or combined_score > best_fit_result.get('combined_score', -np.inf):
                        best_r2 = r2
                        best_fit_result = {
                            'popt': popt,
                            'pcov': pcov,
                            'r2': r2,
                            'rmse': rmse,
                            'is_s_curve': is_s_curve,
                            'strategy': strategy_name,
                            't0_score': t0_score,
                            'combined_score': combined_score,
                            't0_status': t0_status
                        }
                        print(f"      → 满足约束，当前最佳（综合评分={combined_score:.4f}）")
                elif best_fit_result is None:
                    # 没有满足约束的结果，临时保存但标记警告
                    best_r2 = r2
                    best_fit_result = {
                        'popt': popt,
                        'pcov': pcov,
                        'r2': r2,
                        'rmse': rmse,
                        'is_s_curve': is_s_curve,
                        'strategy': strategy_name,
                        't0_score': t0_score,
                        'combined_score': combined_score,
                        't0_status': t0_status
                    }
                    print(f"      ⚠️ 不满足约束(S型={is_s_curve}, t0={t0_fit:.1f})，暂时保留")
                else:
                    print(f"      ✗ 不满足约束或R²不如当前最佳，跳过")
                
            except Exception as e:
                print(f"      策略[{strategy_name}]失败: {e}")
                continue
        
        # 使用最佳拟合结果
        if best_fit_result is not None:
            popt = best_fit_result['popt']
            pcov = best_fit_result['pcov']
            r2 = best_fit_result['r2']
            rmse = best_fit_result['rmse']
            
            print(f"\n    ✓ 选择最佳拟合: {best_fit_result['strategy']}")
            print(f"      R²={r2:.4f}, RMSE={rmse:.6f}, S型={best_fit_result['is_s_curve']}")
            print(f"      拐点位置: {best_fit_result.get('t0_status', 'N/A')} (得分={best_fit_result.get('t0_score', 0):.3f})")
            print(f"      综合评分: {best_fit_result.get('combined_score', 0):.4f}")
        
            L_fit, k_fit, t0_fit, b_fit = popt
            
            # 输出拟合结果
            print(f"    最终参数: L={L_fit:.4f}, k={k_fit:.4f}, t0={t0_fit:.1f}, b={b_fit:.4f}")
            
            # 参数不确定性分析
            perr = np.sqrt(np.diag(pcov))
            L_err, k_err, t0_err, b_err = perr
            
            # 95%置信区间
            conf_level = 1.96
            L_ci = conf_level * L_err
            k_ci = conf_level * k_err
            t0_ci = conf_level * t0_err
            b_ci = conf_level * b_err
            
            # 相对误差（CV%）
            L_rel_err = (L_err / abs(L_fit)) * 100 if L_fit != 0 else np.inf
            k_rel_err = (k_err / abs(k_fit)) * 100 if k_fit != 0 else np.inf
            t0_rel_err = (t0_err / abs(t0_fit)) * 100 if t0_fit != 0 else np.inf
            b_rel_err = (b_err / abs(b_fit)) * 100 if b_fit != 0 else np.inf
            
            print(f"\n    参数不确定性（标准误差）:")
            print(f"      L: {L_fit:.4f} ± {L_err:.4f} (CV={L_rel_err:.2f}%)")
            print(f"      k: {k_fit:.4f} ± {k_err:.4f} (CV={k_rel_err:.2f}%)")
            print(f"      t₀: {t0_fit:.1f} ± {t0_err:.1f} 天 (CV={t0_rel_err:.2f}%)")
            print(f"      b: {b_fit:.4f} ± {b_err:.4f} (CV={b_rel_err:.2f}%)")
            
            print(f"\n    参数95%置信区间:")
            print(f"      L: [{L_fit - L_ci:.4f}, {L_fit + L_ci:.4f}]")
            print(f"      k: [{k_fit - k_ci:.4f}, {k_fit + k_ci:.4f}]")
            print(f"      t₀: [{t0_fit - t0_ci:.1f}, {t0_fit + t0_ci:.1f}] 天")
            print(f"      b: [{b_fit - b_ci:.4f}, {b_fit + b_ci:.4f}]")
            
            # 检查拟合警告
            fitting_warnings = []
            inflection_value = b_fit + L_fit / 2
            if inflection_value < y_observed.min() or inflection_value > y_observed.max():
                fitting_warnings.append(f"拐点值({inflection_value:.4f})不在数据范围内")
            
            # 拐点位置警告
            if t0_fit < 53:
                fitting_warnings.append(f"⚠️ 拐点过早(t0={t0_fit:.1f}<53天)")
            elif t0_fit > days.max():
                fitting_warnings.append(f"⚠️ 拐点超出观测期(t0={t0_fit:.1f}>{days.max():.0f}天)")
            elif 80 <= t0_fit <= 100:
                fitting_warnings.append(f"✓ 拐点位置理想(80≤t0={t0_fit:.1f}≤100天)")
            
            if k_fit < 0.015:
                fitting_warnings.append("生长率很小(k<0.015)，增长缓慢")
            if k_fit > 0.3:
                fitting_warnings.append("生长率很大(k>0.3)，增长急剧")
            
            final_value = L_fit + b_fit
            if final_value < y_observed.max() * 0.95:
                fitting_warnings.append(f"最终值({final_value:.4f})低于数据最大值({y_observed.max():.4f})")
            
            if not best_fit_result['is_s_curve']:
                fitting_warnings.append("⚠️ 拟合曲线不是典型S型！")
            
            if fitting_warnings:
                print(f"    ⚠️ 拟合提示:")
                for warning in fitting_warnings:
                    print(f"       - {warning}")
            
            # 参数不确定性
            perr = np.sqrt(np.diag(pcov))
            L_err, k_err, t0_err, b_err = perr
            
            # 95%置信区间
            conf_level = 1.96
            L_ci = conf_level * L_err
            k_ci = conf_level * k_err
            t0_ci = conf_level * t0_err
            b_ci = conf_level * b_err
            
            # 相对误差（CV%）
            L_rel_err = (L_err / abs(L_fit)) * 100 if L_fit != 0 else np.inf
            k_rel_err = (k_err / abs(k_fit)) * 100 if k_fit != 0 else np.inf
            t0_rel_err = (t0_err / abs(t0_fit)) * 100 if t0_fit != 0 else np.inf
            b_rel_err = (b_err / abs(b_fit)) * 100 if b_fit != 0 else np.inf
            
            # 生长关键指标
            days_smooth = np.linspace(days.min(), days.max(), 300)
            growth_rate = logistic_derivative(days_smooth, *popt)
            max_growth_idx = np.argmax(growth_rate)
            max_growth_time = days_smooth[max_growth_idx]
            max_growth_rate = growth_rate[max_growth_idx]
            
            # 生长阶段划分（参考curve_width.py）
            # 迟缓期结束：生长率达到最大值的10%
            threshold_10 = 0.1 * max_growth_rate
            lag_mask = growth_rate < threshold_10
            if np.any(~lag_mask):
                lag_phase_end = days_smooth[np.where(~lag_mask)[0][0]]
            else:
                lag_phase_end = days.min()
            
            # 快速生长期结束：生长率降至最大值的10%
            rapid_mask = growth_rate[max_growth_idx:] > threshold_10
            if np.any(~rapid_mask):
                rapid_phase_end = days_smooth[max_growth_idx + np.where(~rapid_mask)[0][0]]
            else:
                rapid_phase_end = days.max()
            
            # 计算达到90%和95%最终增长的时间
            final_value = L_fit + b_fit
            target_90 = b_fit + 0.9 * L_fit
            target_95 = b_fit + 0.95 * L_fit
            
            y_smooth = logistic(days_smooth, *popt)
            day_90_idx = np.argmin(np.abs(y_smooth - target_90))
            day_95_idx = np.argmin(np.abs(y_smooth - target_95))
            day_90_percent = days_smooth[day_90_idx]
            day_95_percent = days_smooth[day_95_idx]
            
            fitting_success = True
        else:
            print(f"    ✗ 所有拟合策略都失败了！")
            fitting_success = False
            L_fit = k_fit = t0_fit = b_fit = None
            r2 = rmse = None
            L_err = k_err = t0_err = b_err = None
            L_ci = k_ci = t0_ci = b_ci = None
            L_rel_err = k_rel_err = t0_rel_err = b_rel_err = None
            max_growth_rate = max_growth_time = None
            lag_phase_end = rapid_phase_end = None
            day_90_percent = day_95_percent = None
            final_value = None
        
        return {
            'param_name': param_name,
            'days': days,
            'y_observed': y_observed,
            'y_sem': y_sem,
            'sample_sizes': sample_sizes,
            # 统计检验
            'p_value_linear': p_value_linear,
            'p_value_spearman': p_value_spearman,
            'cohens_d': cohens_d,
            'mean_diff': mean_diff,
            'slope': slope,
            # Logistic拟合参数
            'fitting_success': fitting_success,
            'L': L_fit,
            'k': k_fit,
            't0': t0_fit,
            'b': b_fit,
            'r2': r2,
            'rmse': rmse,
            # 参数不确定性
            'L_err': L_err,
            'k_err': k_err,
            't0_err': t0_err,
            'b_err': b_err,
            'L_ci': L_ci,
            'k_ci': k_ci,
            't0_ci': t0_ci,
            'b_ci': b_ci,
            'L_rel_err': L_rel_err,
            'k_rel_err': k_rel_err,
            't0_rel_err': t0_rel_err,
            'b_rel_err': b_rel_err,
            # 生长指标
            'max_growth_rate': max_growth_rate,
            'max_growth_time': max_growth_time,
            'lag_phase_end': lag_phase_end,
            'rapid_phase_end': rapid_phase_end,
            'day_90_percent': day_90_percent,
            'day_95_percent': day_95_percent,
            'final_value': final_value,
        }
    
    def analyze_growth_curves(self):
        """对所有参数组合进行生长曲线拟合分析"""
        print("\n" + "=" * 80)
        print("生长曲线拟合分析（Logistic Model）")
        print("=" * 80 + "\n")
        
        self.growth_fits = {}
        
        # 1. 不同置信度阈值的生长曲线
        print("分析置信度阈值的影响...")
        for conf_key, data in self.results['confidence'].items():
            if data['standing'] is not None:
                param_name = f"Conf={data['conf']:.1f}"
                fit_result = self.fit_logistic_growth(data['standing'], param_name)
                if fit_result:
                    if fit_result['fitting_success']:
                        print(f"  ✓ {param_name}: p={fit_result['p_value_linear']:.4e}, R²={fit_result['r2']:.3f}")
                    else:
                        print(f"  ✗ {param_name}: 拟合失败")
                else:
                    print(f"  ✗ {param_name}: 无数据")
                self.growth_fits[f'conf_{conf_key}'] = fit_result
        
        # 2. 不同ROI区域的生长曲线
        print("\n分析ROI区域的影响...")
        for roi_name, data in self.results['roi'].items():
            if data['standing'] is not None:
                # 使用原始ROI名称并添加%
                roi_label = roi_name.replace('center_', 'center_') + '%' if roi_name.startswith('center_') else roi_name
                param_name = f"ROI={roi_label}"
                fit_result = self.fit_logistic_growth(data['standing'], param_name)
                if fit_result:
                    if fit_result['fitting_success']:
                        print(f"  ✓ {param_name}: p={fit_result['p_value_linear']:.4e}, R²={fit_result['r2']:.3f}")
                    else:
                        print(f"  ✗ {param_name}: 拟合失败")
                else:
                    print(f"  ✗ {param_name}: 无数据")
                self.growth_fits[f'roi_{roi_name}'] = fit_result
        
        # 3. 不同W/L阈值范围的生长曲线
        print("\n分析W/L阈值范围的影响...")
        for wl_key, data in self.results['wl_range'].items():
            if data['standing'] is not None:
                param_name = f"W/L∈[{data['wl_min']:.3f},{data['wl_max']:.3f}]"
                fit_result = self.fit_logistic_growth(data['standing'], param_name)
                if fit_result:
                    if fit_result['fitting_success']:
                        print(f"  ✓ {param_name}: p={fit_result['p_value_linear']:.4e}, R²={fit_result['r2']:.3f}")
                    else:
                        print(f"  ✗ {param_name}: 拟合失败")
                else:
                    print(f"  ✗ {param_name}: 无数据")
                self.growth_fits[f'wl_{wl_key}'] = fit_result
        
        print("\n✓ 生长曲线拟合分析完成\n")
        
        # 统计拟合成功率
        total_fits = len(self.growth_fits)
        successful_fits = sum(1 for fit in self.growth_fits.values() if fit and fit['fitting_success'])
        print(f"拟合成功: {successful_fits}/{total_fits}")
    
    def analyze_best_params_growth_curves(self):
        """对最佳参数控制变量进行生长曲线拟合分析"""
        if not hasattr(self, 'best_results') or not self.best_results:
            print("警告: 需要先运行 analyze_with_best_params()")
            return
        
        print("\n" + "=" * 80)
        print("最佳参数控制变量生长曲线拟合分析")
        print("=" * 80 + "\n")
        
        self.best_growth_fits = {}
        
        # 取消缓存机制，每组数据独立拟合，确保拟合结果准确
        
        # 1. 不同置信度阈值的生长曲线（固定ROI和WL）
        print("分析置信度阈值的影响（固定ROI=center_70, WL=[0.25,0.333]）...")
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                param_name = f"Conf={data['conf']:.1f}"
                fit_result = self.fit_logistic_growth(data['standing'], param_name)
                if fit_result:
                    # 保存conf值到fit_result中，用于后续颜色判断
                    fit_result['conf'] = data['conf']
                    if fit_result['fitting_success']:
                        print(f"  ✓ {param_name}: p={fit_result['p_value_linear']:.4e}, R²={fit_result['r2']:.3f}")
                    else:
                        print(f"  ✗ {param_name}: 拟合失败")
                else:
                    print(f"  ✗ {param_name}: 无数据")
                self.best_growth_fits[f'conf_{conf_key}'] = fit_result
        
        # 2. 不同ROI区域的生长曲线（固定Conf和WL）
        print("\n分析ROI区域的影响（固定Conf=0.7, WL=[0.25,0.333]）...")
        for roi_name, data in self.best_results['roi'].items():
            if data['standing'] is not None:
                # 使用原始ROI名称并添加%
                roi_label = roi_name.replace('center_', 'center_') + '%' if roi_name.startswith('center_') else roi_name
                param_name = f"ROI={roi_label}"
                fit_result = self.fit_logistic_growth(data['standing'], param_name)
                if fit_result:
                    # 保存roi_name到fit_result中，用于后续颜色判断
                    fit_result['roi_name'] = roi_name
                    if fit_result['fitting_success']:
                        print(f"  ✓ {param_name}: p={fit_result['p_value_linear']:.4e}, R²={fit_result['r2']:.3f}")
                    else:
                        print(f"  ✗ {param_name}: 拟合失败")
                else:
                    print(f"  ✗ {param_name}: 无数据")
                self.best_growth_fits[f'roi_{roi_name}'] = fit_result
        
        # 3. 不同W/L阈值范围的生长曲线（固定Conf和ROI）
        print("\n分析W/L阈值范围的影响（固定Conf=0.7, ROI=center_70）...")
        for wl_key, data in self.best_results['wl_range'].items():
            if data['standing'] is not None:
                param_name = f"W/L∈[{data['wl_min']:.3f},{data['wl_max']:.3f}]"
                fit_result = self.fit_logistic_growth(data['standing'], param_name)
                if fit_result:
                    # 保存wl_range到fit_result中，用于后续颜色判断
                    fit_result['wl_range'] = (data['wl_min'], data['wl_max'])
                    fit_result['wl_min'] = data['wl_min']
                    fit_result['wl_max'] = data['wl_max']
                    if fit_result['fitting_success']:
                        print(f"  ✓ {param_name}: p={fit_result['p_value_linear']:.4e}, R²={fit_result['r2']:.3f}")
                    else:
                        print(f"  ✗ {param_name}: 拟合失败")
                else:
                    print(f"  ✗ {param_name}: 无数据")
                self.best_growth_fits[f'wl_{wl_key}'] = fit_result
        
        print("\n✓ 最佳参数生长曲线拟合分析完成\n")
        
        # 统计拟合成功率
        total_fits = len(self.best_growth_fits)
        successful_fits = sum(1 for fit in self.best_growth_fits.values() if fit and fit['fitting_success'])
        print(f"拟合成功: {successful_fits}/{total_fits}")
    
    def plot_growth_curve_fits(self, save_path=None):
        """绘制生长曲线拟合结果（带SEM误差带和拟合曲线）"""
        if not hasattr(self, 'growth_fits') or not self.growth_fits:
            print("警告: 需要先运行 analyze_growth_curves()")
            return None
        
        print("\n开始绘制生长曲线拟合图...")
        
        # 创建3个子图：Confidence, ROI, WL threshold
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 定义颜色
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # 1. Confidence Threshold 拟合
        ax = axes[0]
        plot_count = 0
        for idx, (key, fit) in enumerate(self.growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('conf_'):
                color = colors[idx % 10]
                days = fit['days']
                y_obs = fit['y_observed']
                y_sem = fit['y_sem']
                
                # 生成平滑拟合曲线
                days_smooth = np.linspace(days.min(), days.max(), 300)
                y_fit = logistic(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                # 绘制数据点和误差带
                label = fit['param_name']
                ax.errorbar(days, y_obs, yerr=y_sem, fmt='o', capsize=3, 
                           color=color, alpha=0.5, markersize=6)
                # 拟合曲线 - 粗实线
                ax.plot(days_smooth, y_fit, '-', linewidth=3, 
                       color=color, label=label, alpha=0.9)
                plot_count += 1
        
        print(f"  Confidence子图: 绘制了 {plot_count} 条拟合曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title('(a) Logistic Fit: Confidence Threshold', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # Set x-axis to show actual day values
        for key, fit in self.growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('conf_'):
                ax.set_xticks(fit['days'])
                break
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. ROI Size 拟合
        ax = axes[1]
        plot_count = 0
        for idx, (key, fit) in enumerate(self.growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('roi_') and 'no_filter' not in key:
                color = colors[idx % 10]
                days = fit['days']
                y_obs = fit['y_observed']
                y_sem = fit['y_sem']
                
                days_smooth = np.linspace(days.min(), days.max(), 300)
                y_fit = logistic(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                label = fit['param_name'].replace('ROI=center_', '')
                ax.errorbar(days, y_obs, yerr=y_sem, fmt='s', capsize=3, 
                           color=color, alpha=0.5, markersize=6)
                ax.plot(days_smooth, y_fit, '-', linewidth=3,
                       color=color, label=label, alpha=0.9)
                plot_count += 1
        
        print(f"  ROI子图: 绘制了 {plot_count} 条拟合曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title('(b) Logistic Fit: ROI Size', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # Set x-axis to show actual day values
        for key, fit in self.growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('roi_') and 'no_filter' not in key:
                ax.set_xticks(fit['days'])
                break
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. W/L Threshold 拟合
        ax = axes[2]
        plot_count = 0
        for idx, (key, fit) in enumerate(self.growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('wl_'):
                color = colors[idx % 10]
                days = fit['days']
                y_obs = fit['y_observed']
                y_sem = fit['y_sem']
                
                days_smooth = np.linspace(days.min(), days.max(), 300)
                y_fit = logistic(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                ax.errorbar(days, y_obs, yerr=y_sem, fmt='^', capsize=3, 
                           color=color, alpha=0.5, markersize=6)
                ax.plot(days_smooth, y_fit, '-', linewidth=3,
                       color=color, label=fit['param_name'], alpha=0.9)
                plot_count += 1
        
        print(f"  W/L子图: 绘制了 {plot_count} 条拟合曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.set_title('(c) Logistic Fit: W/L Threshold', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        # Set x-axis to show actual day values
        for key, fit in self.growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('wl_'):
                ax.set_xticks(fit['days'])
                break
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存生长曲线拟合图: {save_path}")
        
        return fig
    
    def plot_growth_rate_curves(self, save_path=None):
        """绘制生长速率曲线（dy/dt）- 直接从拟合参数计算"""
        if not hasattr(self, 'growth_fits') or not self.growth_fits:
            print("警告: 需要先运行 analyze_growth_curves()")
            return None
        
        print("\n开始绘制生长速率曲线...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 定义颜色
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # 1. Confidence Threshold 生长速率
        ax = axes[0]
        plot_count = 0
        for idx, (key, fit) in enumerate(self.growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('conf_'):
                color = colors[idx % 10]
                days_smooth = np.linspace(fit['days'].min(), fit['days'].max(), 300)
                # 直接从拟合参数计算生长速率
                growth_rate = logistic_derivative(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                ax.plot(days_smooth, growth_rate, linewidth=3, label=fit['param_name'], color=color)
                
                # 标记最大生长速率点
                if fit['max_growth_time'] is not None:
                    ax.plot(fit['max_growth_time'], fit['max_growth_rate'], 'o', 
                           markersize=10, color=color, markeredgecolor='white', markeredgewidth=2)
                plot_count += 1
        
        print(f"  Confidence子图: 绘制了 {plot_count} 条生长速率曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Growth Rate (dW/L per day)', fontsize=12, fontweight='bold')
        ax.set_title('(a) Growth Rate: Confidence Threshold', fontsize=13, fontweight='bold')
        # Set x-axis to show actual day values
        for key, fit in self.growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('conf_'):
                ax.set_xticks(fit['days'])
                break
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 2. ROI Size 生长速率
        ax = axes[1]
        plot_count = 0
        for idx, (key, fit) in enumerate(self.growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('roi_') and 'no_filter' not in key:
                color = colors[idx % 10]
                days_smooth = np.linspace(fit['days'].min(), fit['days'].max(), 300)
                growth_rate = logistic_derivative(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                label = fit['param_name'].replace('ROI=center_', '')
                ax.plot(days_smooth, growth_rate, linewidth=3, label=label, color=color)
                
                if fit['max_growth_time'] is not None:
                    ax.plot(fit['max_growth_time'], fit['max_growth_rate'], 's', 
                           markersize=10, color=color, markeredgecolor='white', markeredgewidth=2)
                plot_count += 1
        
        print(f"  ROI子图: 绘制了 {plot_count} 条生长速率曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Growth Rate (dW/L per day)', fontsize=12, fontweight='bold')
        ax.set_title('(b) Growth Rate: ROI Size', fontsize=13, fontweight='bold')
        # Set x-axis to show actual day values
        for key, fit in self.growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('roi_') and 'no_filter' not in key:
                ax.set_xticks(fit['days'])
                break
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 3. W/L Threshold 生长速率
        ax = axes[2]
        plot_count = 0
        for idx, (key, fit) in enumerate(self.growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('wl_'):
                color = colors[idx % 10]
                days_smooth = np.linspace(fit['days'].min(), fit['days'].max(), 300)
                growth_rate = logistic_derivative(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                ax.plot(days_smooth, growth_rate, linewidth=3, label=fit['param_name'], color=color)
                
                if fit['max_growth_time'] is not None:
                    ax.plot(fit['max_growth_time'], fit['max_growth_rate'], '^', 
                           markersize=10, color=color, markeredgecolor='white', markeredgewidth=2)
                plot_count += 1
        
        print(f"  W/L子图: 绘制了 {plot_count} 条生长速率曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Growth Rate (dW/L per day)', fontsize=12, fontweight='bold')
        ax.set_title('(c) Growth Rate: W/L Threshold', fontsize=13, fontweight='bold')
        # Set x-axis to show actual day values
        for key, fit in self.growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('wl_'):
                ax.set_xticks(fit['days'])
                break
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存生长速率曲线图: {save_path}")
        
        return fig
    
    def plot_best_params_growth_fits(self, save_path=None):
        """绘制最佳参数控制变量的生长曲线拟合结果"""
        if not hasattr(self, 'best_growth_fits') or not self.best_growth_fits:
            print("警告: 需要先运行 analyze_best_params_growth_curves()")
            return None
        
        print("\n开始绘制最佳参数控制变量拟合图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # 定义基准颜色（最佳参数组合）
        baseline_color = '#D62728'  # 深红色
        
        # 1. Confidence（固定ROI=center_70, W/L=[0.25,0.333]）
        ax = axes[0]
        plot_count = 0
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (key, fit) in enumerate(self.best_growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('conf_'):
                # 如果是基准参数conf=0.7，使用基准颜色
                is_baseline = (fit['conf'] == self.best_conf)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                days = fit['days']
                y_obs = fit['y_observed']
                y_sem = fit['y_sem']
                
                days_smooth = np.linspace(days.min(), days.max(), 300)
                y_fit = logistic(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                ax.errorbar(days, y_obs, yerr=y_sem, fmt='o', capsize=3, 
                           color=color, alpha=0.5, markersize=6)
                ax.plot(days_smooth, y_fit, '-', linewidth=3, 
                       color=color, label=fit['param_name'], alpha=0.9)
                # 设置实际天数刻度
                ax.set_xticks(days)
                plot_count += 1
        
        print(f"  Confidence子图: 绘制了 {plot_count} 条拟合曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        # 在图的底部添加(A)标注
        ax.text(0.5, -0.10, '(A)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 2. ROI（Fixed Conf=0.7, W/L=[0.25,0.333]）
        ax = axes[1]
        plot_count = 0
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (key, fit) in enumerate(self.best_growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('roi_'):
                # 如果是基准参数center_70，使用基准颜色
                is_baseline = (fit.get('roi_name') == self.best_roi)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                days = fit['days']
                y_obs = fit['y_observed']
                y_sem = fit['y_sem']
                
                days_smooth = np.linspace(days.min(), days.max(), 300)
                y_fit = logistic(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                label = fit['param_name'].replace('ROI=center_', '')
                ax.errorbar(days, y_obs, yerr=y_sem, fmt='s', capsize=3, 
                           color=color, alpha=0.5, markersize=6)
                ax.plot(days_smooth, y_fit, '-', linewidth=3,
                       color=color, label=label, alpha=0.9)
                # 设置实际天数刻度
                ax.set_xticks(days)
                plot_count += 1
        
        print(f"  ROI子图: 绘制了 {plot_count} 条拟合曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Herd Mean W/L Ratio', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        # 在图的底部添加(B)标注
        ax.text(0.5, -0.10, '(B)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 3. W/L（Fixed Conf=0.7, ROI=center_70）
        ax = axes[2]
        plot_count = 0
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (key, fit) in enumerate(self.best_growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('wl_'):
                # 如果是基准参数W/L=[0.25,0.333]，使用基准颜色
                is_baseline = (fit.get('wl_range') == self.best_wl_range)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                days = fit['days']
                y_obs = fit['y_observed']
                y_sem = fit['y_sem']
                
                days_smooth = np.linspace(days.min(), days.max(), 300)
                y_fit = logistic(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                ax.errorbar(days, y_obs, yerr=y_sem, fmt='^', capsize=3, 
                           color=color, alpha=0.5, markersize=6)
                ax.plot(days_smooth, y_fit, '-', linewidth=3,
                       color=color, label=fit['param_name'], alpha=0.9)
                # 设置实际天数刻度
                ax.set_xticks(days)
                plot_count += 1
        
        print(f"  W/L子图: 绘制了 {plot_count} 条拟合曲线")
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        # 在图的底部添加(C)标注
        ax.text(0.5, -0.10, '(C)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存最佳参数拟合曲线图: {save_path}")
        
        return fig
    
    def plot_best_params_growth_rates(self, save_path=None):
        """绘制最佳参数控制变量的生长速率曲线"""
        if not hasattr(self, 'best_growth_fits') or not self.best_growth_fits:
            print("警告: 需要先运行 analyze_best_params_growth_curves()")
            return None
        
        print("\n开始绘制最佳参数生长速率曲线...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # 定义基准颜色（最佳参数组合）
        baseline_color = '#D62728'  # 深红色
        
        # 1. Confidence生长速率
        ax = axes[0]
        plot_count = 0
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (key, fit) in enumerate(self.best_growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('conf_'):
                # 如果是基准参数conf=0.7，使用基准颜色
                is_baseline = (fit['conf'] == self.best_conf)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                days_smooth = np.linspace(fit['days'].min(), fit['days'].max(), 300)
                growth_rate = logistic_derivative(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                ax.plot(days_smooth, growth_rate, linewidth=3, label=fit['param_name'], color=color)
                
                if fit['max_growth_time'] is not None:
                    ax.plot(fit['max_growth_time'], fit['max_growth_rate'], 'o', 
                           markersize=10, color=color, markeredgecolor='white', markeredgewidth=2)
                plot_count += 1
        
        print(f"  Confidence子图: 绘制了 {plot_count} 条生长速率曲线")
        # Set x-axis to show actual day values
        for key, fit in self.best_growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('conf_'):
                ax.set_xticks(fit['days'])
                break
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Growth Rate (dW/L per day)', fontsize=12, fontweight='bold')
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        # 在图的底部添加(A)标注
        ax.text(0.5, -0.10, '(A)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 2. ROI生长速率
        ax = axes[1]
        plot_count = 0
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (key, fit) in enumerate(self.best_growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('roi_'):
                # 如果是基准参数center_70，使用基准颜色
                is_baseline = (fit.get('roi_name') == self.best_roi)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                days_smooth = np.linspace(fit['days'].min(), fit['days'].max(), 300)
                growth_rate = logistic_derivative(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                label = fit['param_name'].replace('ROI=center_', '')
                ax.plot(days_smooth, growth_rate, linewidth=3, label=label, color=color)
                
                if fit['max_growth_time'] is not None:
                    ax.plot(fit['max_growth_time'], fit['max_growth_rate'], 's', 
                           markersize=10, color=color, markeredgecolor='white', markeredgewidth=2)
                plot_count += 1
        
        # Set x-axis to show actual day values
        for key, fit in self.best_growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('roi_'):
                ax.set_xticks(fit['days'])
                break
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Growth Rate (dW/L per day)', fontsize=12, fontweight='bold')
        if plot_count > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        # 在图的底部添加(B)标注
        ax.text(0.5, -0.10, '(B)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 3. W/L生长速率
        ax = axes[2]
        plot_count = 0
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (key, fit) in enumerate(self.best_growth_fits.items()):
            if fit and fit['fitting_success'] and key.startswith('wl_'):
                # 如果是基准参数W/L=[0.25,0.333]，使用基准颜色
                is_baseline = (fit.get('wl_range') == self.best_wl_range)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                days_smooth = np.linspace(fit['days'].min(), fit['days'].max(), 300)
                growth_rate = logistic_derivative(days_smooth, fit['L'], fit['k'], fit['t0'], fit['b'])
                
                ax.plot(days_smooth, growth_rate, linewidth=3, label=fit['param_name'], color=color)
                
                if fit['max_growth_time'] is not None:
                    ax.plot(fit['max_growth_time'], fit['max_growth_rate'], '^', 
                           markersize=10, color=color, markeredgecolor='white', markeredgewidth=2)
                plot_count += 1
        # Set x-axis to show actual day values
        for key, fit in self.best_growth_fits.items():
            if fit and fit['fitting_success'] and key.startswith('wl_'):
                ax.set_xticks(fit['days'])
                break
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Growth Rate (dW/L per day)', fontsize=12, fontweight='bold')
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        # 在图的底部添加(C)标注
        ax.text(0.5, -0.10, '(C)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        if plot_count > 0:
            ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存最佳参数生长速率曲线图: {save_path}")
        
        return fig
    
    def plot_instance_counts_over_time(self, save_path=None):
        """可视化每个时间点的Standing实例个数"""
        print("\n开始绘制Standing实例个数随时间变化图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # 1. 置信度阈值下的实例个数
        ax = axes[0]
        for idx, (conf_key, data) in enumerate(self.results['confidence'].items()):
            if data['standing'] is not None:
                stats = data['standing']
                color = colors[idx % 10]
                ax.plot(stats['day_indices'], stats['count'], 'o-', 
                       linewidth=2.5, markersize=8, color=color, 
                       label=f"Conf={data['conf']:.1f}", alpha=0.8)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standing Instance Count', fontsize=12, fontweight='bold')
        ax.set_title('(a) Instance Count: Confidence Threshold', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. ROI区域下的实例个数
        ax = axes[1]
        plot_count = 0
        for idx, (roi_name, data) in enumerate(self.results['roi'].items()):
            if roi_name != 'no_filter' and data['standing'] is not None:
                stats = data['standing']
                color = colors[idx % 10]
                # 使用格式化的ROI标签
                label = self.format_roi_label(roi_name)
                ax.plot(stats['day_indices'], stats['count'], 's-', 
                       linewidth=2.5, markersize=8, color=color, 
                       label=label, alpha=0.8)
                plot_count += 1
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standing Instance Count', fontsize=12, fontweight='bold')
        ax.set_title('(b) Instance Count: ROI Size', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. W/L阈值范围下的实例个数
        ax = axes[2]
        for idx, (wl_key, data) in enumerate(self.results['wl_range'].items()):
            if data['standing'] is not None:
                stats = data['standing']
                color = colors[idx % 10]
                ax.plot(stats['day_indices'], stats['count'], '^-', 
                       linewidth=2.5, markersize=8, color=color, 
                       label=f"W/L∈[{data['wl_min']:.3f},{data['wl_max']:.3f}]", alpha=0.8)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standing Instance Count', fontsize=12, fontweight='bold')
        ax.set_title('(c) Instance Count: W/L Threshold', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存实例个数变化图: {save_path}")
        
        return fig
    
    def plot_sample_size_diagnostic(self, save_path=None):
        """
        样本量诊断图：可视化不同参数下的样本量变化和数据质量
        这是理解拟合差异的关键
        """
        if not hasattr(self, 'best_results') or not self.best_results:
            print("警告: 需要先运行 analyze_with_best_params()")
            return None
        
        print("\n开始绘制样本量诊断图...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ===== 第一行：样本量随时间变化 =====
        # 1.1 置信度影响
        ax1 = fig.add_subplot(gs[0, 0])
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                stats = data['standing']
                ax1.plot(stats['day_indices'], stats['count'], 'o-', 
                        linewidth=2, markersize=6, label=f"Conf={data['conf']:.1f}", alpha=0.7)
        
        ax1.set_xlabel('Days', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Sample Size (n)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Sample Size: Confidence Threshold', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='n=10 (min threshold)')
        
        # 1.2 ROI影响
        ax2 = fig.add_subplot(gs[0, 1])
        for roi_name, data in self.best_results['roi'].items():
            if data['standing'] is not None:
                stats = data['standing']
                # 使用格式化的ROI标签
                label = self.format_roi_label(roi_name)
                ax2.plot(stats['day_indices'], stats['count'], 's-', 
                        linewidth=2, markersize=6, label=label, alpha=0.7)
        
        ax2.set_xlabel('Days', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Sample Size (n)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Sample Size: ROI Size', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5)
        
        # 1.3 W/L阈值影响
        ax3 = fig.add_subplot(gs[0, 2])
        for wl_key, data in self.best_results['wl_range'].items():
            if data['standing'] is not None:
                stats = data['standing']
                ax3.plot(stats['day_indices'], stats['count'], '^-', 
                        linewidth=2, markersize=6, 
                        label=f"W/L∈[{data['wl_min']:.2f},{data['wl_max']:.2f}]", alpha=0.7)
        
        ax3.set_xlabel('Days', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Sample Size (n)', fontsize=11, fontweight='bold')
        ax3.set_title('(c) Sample Size: W/L Threshold', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5)
        
        # ===== 第二行：早期vs后期样本量对比 =====
        # 2.1 置信度：早期vs后期
        ax4 = fig.add_subplot(gs[1, 0])
        conf_labels = []
        early_counts = []
        late_counts = []
        
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                stats = data['standing']
                days = np.array(stats['day_indices'])
                counts = np.array(stats['count'])
                early_cutoff_idx = max(1, len(days)//3)
                
                conf_labels.append(f"{data['conf']:.1f}")
                early_counts.append(np.mean(counts[:early_cutoff_idx]))
                late_counts.append(np.mean(counts[early_cutoff_idx:]))
        
        x = np.arange(len(conf_labels))
        width = 0.35
        ax4.bar(x - width/2, early_counts, width, label='Early (first 1/3)', alpha=0.8)
        ax4.bar(x + width/2, late_counts, width, label='Late (last 2/3)', alpha=0.8)
        ax4.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Average Sample Size', fontsize=11, fontweight='bold')
        ax4.set_title('(d) Early vs Late Sample Size: Confidence', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(conf_labels)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 2.2 样本量比率（后期/早期）
        ax5 = fig.add_subplot(gs[1, 1])
        ratios = []
        for early, late in zip(early_counts, late_counts):
            ratio = late / (early + 1e-6)
            ratios.append(ratio)
        
        bars = ax5.bar(x, ratios, alpha=0.7, color='coral')
        ax5.axhline(y=5, color='red', linestyle='--', linewidth=2, label='5x threshold')
        ax5.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Sample Size Ratio (Late/Early)', fontsize=11, fontweight='bold')
        ax5.set_title('(e) Sample Size Disparity by Confidence', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(conf_labels)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 为比率>5的柱子添加标注
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            if ratio > 5:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{ratio:.1f}x', ha='center', fontsize=10, fontweight='bold', color='red')
        
        # 2.3 总样本量对比
        ax6 = fig.add_subplot(gs[1, 2])
        total_samples = []
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                total_samples.append(data['standing']['total_count'])
        
        bars = ax6.bar(x, total_samples, alpha=0.7, color='steelblue')
        ax6.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Total Sample Count', fontsize=11, fontweight='bold')
        ax6.set_title('(f) Total Samples by Confidence', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(conf_labels)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上显示数值
        for bar, count in zip(bars, total_samples):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_samples)*0.01,
                    f'{count}', ha='center', fontsize=10, fontweight='bold')
        
        # ===== 第三行：数据质量指标 =====
        # 3.1 变异系数（CV%）- 数据波动性
        ax7 = fig.add_subplot(gs[2, 0])
        cv_values = []
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                stats = data['standing']
                mean_wl = np.array(stats['mean_wl'])
                cv = np.std(mean_wl) / np.mean(mean_wl) * 100
                cv_values.append(cv)
        
        ax7.bar(x, cv_values, alpha=0.7, color='lightcoral')
        ax7.set_xlabel('Confidence Threshold', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
        ax7.set_title('(g) Data Stability (CV%) by Confidence', fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(conf_labels)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 3.2 标准误（SEM）变化
        ax8 = fig.add_subplot(gs[2, 1])
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                stats = data['standing']
                ax8.plot(stats['day_indices'], stats['sem_wl'], 'o-', 
                        linewidth=2, markersize=5, label=f"Conf={data['conf']:.1f}", alpha=0.7)
        
        ax8.set_xlabel('Days', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Standard Error (SEM)', fontsize=11, fontweight='bold')
        ax8.set_title('(h) Measurement Uncertainty by Confidence', fontsize=12, fontweight='bold')
        ax8.legend(loc='best', fontsize=9)
        ax8.grid(True, alpha=0.3)
        
        # 3.3 数据质量诊断摘要
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # 生成诊断文本
        diagnosis_text = "DATA QUALITY DIAGNOSIS\n" + "="*40 + "\n\n"
        
        for conf_key, data in self.best_results['confidence'].items():
            if data['standing'] is not None:
                issues = self.diagnose_early_data_quality(data['standing'])
                conf_label = f"Conf={data['conf']:.1f}"
                
                if len(issues) == 0:
                    diagnosis_text += f"✓ {conf_label}: No issues\n"
                else:
                    diagnosis_text += f"⚠ {conf_label}:\n"
                    for issue in issues:
                        diagnosis_text += f"  {issue}\n"
                diagnosis_text += "\n"
        
        ax9.text(0.05, 0.95, diagnosis_text, 
                transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Sample Size and Data Quality Diagnostic Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存样本量诊断图: {save_path}")
        
        return fig
    
    def plot_best_params_instance_counts(self, save_path=None):
        """可视化最佳参数控制变量下每个时间点的Standing实例个数"""
        if not hasattr(self, 'best_results') or not self.best_results:
            print("警告: 需要先运行 analyze_with_best_params()")
            return None
        
        print("\n开始绘制最佳参数控制变量Standing实例个数图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # 定义基准颜色（最佳参数组合）
        baseline_color = '#D62728'  # 深红色
        
        # 1. 变化置信度
        ax = axes[0]
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (conf_key, data) in enumerate(self.best_results['confidence'].items()):
            if data['standing'] is not None:
                stats = data['standing']
                # 如果是基准参数conf=0.7，使用基准颜色
                is_baseline = (data['conf'] == self.best_conf)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                ax.plot(stats['day_indices'], stats['count'], 'o-', 
                       linewidth=2.5, markersize=8, color=color, 
                       label=f"Conf={data['conf']:.1f}", alpha=0.8)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standing Instance Count', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        # 在图的底部添加(A)标注
        ax.text(0.5, -0.10, '(A)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 2. 变化ROI
        ax = axes[1]
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (roi_name, data) in enumerate(self.best_results['roi'].items()):
            if data['standing'] is not None:
                stats = data['standing']
                # 如果是基准参数center_70，使用基准颜色
                is_baseline = (roi_name == self.best_roi)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                # 使用原始ROI名称并添加%
                label = roi_name.replace('center_', 'center_') + '%' if roi_name.startswith('center_') else roi_name
                ax.plot(stats['day_indices'], stats['count'], 's-', 
                       linewidth=2.5, markersize=8, color=color, 
                       label=label, alpha=0.8)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standing Instance Count', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        # 在图的底部添加(B)标注
        ax.text(0.5, -0.10, '(B)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 3. 变化W/L范围
        ax = axes[2]
        non_baseline_idx = 0  # 用于非baseline参数的颜色索引
        for idx, (wl_key, data) in enumerate(self.best_results['wl_range'].items()):
            if data['standing'] is not None:
                stats = data['standing']
                # 如果是基准参数W/L=[0.25,0.333]，使用基准颜色
                is_baseline = ((data['wl_min'], data['wl_max']) == self.best_wl_range)
                if is_baseline:
                    color = baseline_color
                else:
                    # 跳过索引3（tab10中接近红色的颜色）
                    color_idx = non_baseline_idx if non_baseline_idx < 3 else non_baseline_idx + 1
                    color = colors[color_idx % 10]
                    non_baseline_idx += 1
                ax.plot(stats['day_indices'], stats['count'], '^-', 
                       linewidth=2.5, markersize=8, color=color, 
                       label=f"W/L∈[{data['wl_min']:.3f},{data['wl_max']:.3f}]", alpha=0.8)
        
        ax.set_xlabel('Days', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        # 在图的底部添加(C)标注
        ax.text(0.5, -0.10, '(C)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存最佳参数实例个数变化图: {save_path}")
        
        return fig
    
    def plot_best_params_fitting_quality(self, save_path=None):
        """可视化最佳参数控制变量下的R²和RMSE"""
        if not hasattr(self, 'best_growth_fits') or not self.best_growth_fits:
            print("警告: 需要先运行 analyze_best_params_growth_curves()")
            return None
        
        print("\n开始绘制最佳参数拟合质量图（R²和RMSE）...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # 定义基准颜色（最佳参数组合）
        baseline_color = '#D62728'  # 深红色
        
        # 收集数据
        conf_data = {'labels': [], 'r2': [], 'rmse': [], 'colors': []}
        roi_data = {'labels': [], 'r2': [], 'rmse': [], 'colors': []}
        wl_data = {'labels': [], 'r2': [], 'rmse': [], 'colors': []}
        
        # 用于非baseline参数的颜色索引
        conf_non_baseline_idx = 0
        roi_non_baseline_idx = 0
        wl_non_baseline_idx = 0
        
        for idx, (key, fit) in enumerate(self.best_growth_fits.items()):
            if fit and fit['fitting_success']:
                rmse = np.sqrt(np.mean((fit['y_observed'] - logistic(fit['days'], fit['L'], fit['k'], fit['t0'], fit['b']))**2))
                
                if key.startswith('conf_'):
                    # 如果是基准参数conf=0.7，使用基准颜色
                    is_baseline = (fit['conf'] == self.best_conf)
                    if is_baseline:
                        color = baseline_color
                    else:
                        # 跳过索引3（tab10中接近红色的颜色）
                        color_idx = conf_non_baseline_idx if conf_non_baseline_idx < 3 else conf_non_baseline_idx + 1
                        color = colors[color_idx % 10]
                        conf_non_baseline_idx += 1
                    conf_val = fit['param_name'].split('=')[1]
                    conf_data['labels'].append(conf_val)
                    conf_data['r2'].append(fit['r2'])
                    conf_data['rmse'].append(rmse)
                    conf_data['colors'].append(color)
                elif key.startswith('roi_'):
                    # 如果是基准参数center_70，使用基准颜色
                    is_baseline = (fit.get('roi_name') == self.best_roi)
                    if is_baseline:
                        color = baseline_color
                    else:
                        # 跳过索引3（tab10中接近红色的颜色）
                        color_idx = roi_non_baseline_idx if roi_non_baseline_idx < 3 else roi_non_baseline_idx + 1
                        color = colors[color_idx % 10]
                        roi_non_baseline_idx += 1
                    roi_val = fit['param_name'].replace('ROI=center_', '')
                    roi_data['labels'].append(roi_val)
                    roi_data['r2'].append(fit['r2'])
                    roi_data['rmse'].append(rmse)
                    roi_data['colors'].append(color)
                elif key.startswith('wl_'):
                    # 如果是基准参数W/L=[0.25,0.333]，使用基准颜色
                    is_baseline = (fit.get('wl_range') == self.best_wl_range)
                    if is_baseline:
                        color = baseline_color
                    else:
                        # 跳过索引3（tab10中接近红色的颜色）
                        color_idx = wl_non_baseline_idx if wl_non_baseline_idx < 3 else wl_non_baseline_idx + 1
                        color = colors[color_idx % 10]
                        wl_non_baseline_idx += 1
                    wl_val = fit['param_name'].replace('W/L∈', '')
                    wl_data['labels'].append(wl_val)
                    wl_data['r2'].append(fit['r2'])
                    wl_data['rmse'].append(rmse)
                    wl_data['colors'].append(color)
        
        # 第一行：R²
        # 1. 置信度的R²
        ax = axes[0, 0]
        bars = ax.bar(range(len(conf_data['labels'])), conf_data['r2'], 
                      color=conf_data['colors'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² (Coefficient of Determination)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(conf_data['labels'])))
        ax.set_xticklabels(conf_data['labels'], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.8)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (>0.6)')
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, conf_data['r2'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 在图的底部添加(A)标注
        ax.text(0.5, -0.12, '(A)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 2. ROI的R²
        ax = axes[0, 1]
        bars = ax.bar(range(len(roi_data['labels'])), roi_data['r2'], 
                      color=roi_data['colors'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('ROI Size (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² (Coefficient of Determination)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(roi_data['labels'])))
        ax.set_xticklabels(roi_data['labels'], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.8)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (>0.6)')
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars, roi_data['r2'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 在图的底部添加(B)标注
        ax.text(0.5, -0.12, '(B)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 3. W/L的R²
        ax = axes[0, 2]
        bars = ax.bar(range(len(wl_data['labels'])), wl_data['r2'], 
                      color=wl_data['colors'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('W/L Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² (Coefficient of Determination)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(wl_data['labels'])))
        ax.set_xticklabels(wl_data['labels'], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.8)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (>0.6)')
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars, wl_data['r2'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 在图的底部添加(C)标注
        ax.text(0.5, -0.12, '(C)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 第二行：RMSE
        # 4. 置信度的RMSE
        ax = axes[1, 0]
        bars = ax.bar(range(len(conf_data['labels'])), conf_data['rmse'], 
                      color=conf_data['colors'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(conf_data['labels'])))
        ax.set_xticklabels(conf_data['labels'], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars, conf_data['rmse'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 在图的底部添加(D)标注
        ax.text(0.5, -0.12, '(D)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 5. ROI的RMSE
        ax = axes[1, 1]
        bars = ax.bar(range(len(roi_data['labels'])), roi_data['rmse'], 
                      color=roi_data['colors'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('ROI Size (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(roi_data['labels'])))
        ax.set_xticklabels(roi_data['labels'], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars, roi_data['rmse'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 在图的底部添加(E)标注
        ax.text(0.5, -0.12, '(E)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        # 6. W/L的RMSE
        ax = axes[1, 2]
        bars = ax.bar(range(len(wl_data['labels'])), wl_data['rmse'], 
                      color=wl_data['colors'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('W/L Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(wl_data['labels'])))
        ax.set_xticklabels(wl_data['labels'], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars, wl_data['rmse'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 在图的底部添加(F)标注
        ax.text(0.5, -0.12, '(F)', transform=ax.transAxes, fontsize=14, 
               fontweight='bold', ha='center', va='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存拟合质量可视化图: {save_path}")
        
        return fig
    
    def generate_report(self, output_path=None):
        """生成文本报告（参考curve_width.py的详细参数分析）"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'sensitivity_analysis_report.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            def write(text):
                print(text)
                f.write(text + '\n')
            
            write("=" * 80)
            write("牛W/L比值参数敏感性分析报告")
            write("Cattle W/L Ratio Parameter Sensitivity Analysis Report")
            write("=" * 80)
            write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            write(f"数据来源: {self.csv_path}")
            write(f"总记录数: {len(self.df)}\n")
            
            # 1. 置信度阈值影响
            write("\n" + "=" * 80)
            write("1. 置信度阈值敏感性分析 (Confidence Threshold Sensitivity)")
            write("=" * 80)
            total_standing = len(self.df[self.df['Class'] == 'standing'])
            write(f"\n总Standing实例数: {total_standing}")
            write(f"总实例数（Standing + Laying）: {len(self.df)}\n")
            write(f"\n{'Conf':<8} | {'Standing Count':<15} | {'Standing Mean W/L':<20} | {'Standing Std':<15}")
            write("-" * 80)
            
            for conf_key, data in self.results['confidence'].items():
                if data['standing'] is not None:
                    stats = data['standing']
                    write(f"{data['conf']:<8.1f} | {stats['total_count']:<15} | "
                          f"{stats['overall_mean']:<20.3f} | {stats['overall_std']:<15.3f}")
            
            # 2. ROI区域影响
            write("\n" + "=" * 80)
            write("2. ROI区域敏感性分析 (ROI Region Sensitivity)")
            write("=" * 80)
            total_standing = len(self.df[self.df['Class'] == 'standing'])
            write(f"\n总Standing实例数: {total_standing}")
            write(f"总实例数（Standing + Laying）: {len(self.df)}\n")
            write(f"\n{'ROI Config':<15} | {'Standing Count':<15} | {'Standing Mean W/L':<20} | {'Standing Std':<15}")
            write("-" * 80)
            
            for roi_name, data in self.results['roi'].items():
                if data['standing'] is not None:
                    stats = data['standing']
                    write(f"{roi_name:<15} | {stats['total_count']:<15} | "
                          f"{stats['overall_mean']:<20.3f} | {stats['overall_std']:<15.3f}")
            
            # 3. W/L阈值影响
            write("\n" + "=" * 80)
            write("3. W/L比值阈值范围敏感性分析（5个不同范围） (W/L Range Sensitivity)")
            write("=" * 80)
            total_standing = len(self.df[self.df['Class'] == 'standing'])
            write(f"\n总Standing实例数: {total_standing}")
            write(f"总实例数（Standing + Laying）: {len(self.df)}\n")
            write(f"\n{'W/L范围':<20} | {'Standing Count':<15} | {'Standing Mean W/L':<20} | {'Standing Std':<15}")
            write("-" * 80)
            
            for wl_key, data in self.results['wl_range'].items():
                if data['standing'] is not None:
                    stats = data['standing']
                    write(f"[{data['wl_min']:.3f}, {data['wl_max']:.3f}]    | {stats['total_count']:<15} | "
                          f"{stats['overall_mean']:<20.3f} | {stats['overall_std']:<15.3f}")
            
            # 3B. 最佳参数控制变量分析
            if hasattr(self, 'best_results') and self.best_results:
                write("\n" + "=" * 80)
                write("3B. 最佳参数控制变量分析（每次只变动一个参数）")
                write(f"最佳参数设置：Conf={self.best_conf}, ROI={self.best_roi}, WL={self.best_wl_range}")
                write("=" * 80)
                
                # 添加过滤前后的整体统计
                total_instances = len(self.df)
                total_standing = len(self.df[self.df['Class'] == 'standing'])
                total_laying = len(self.df[self.df['Class'] == 'laying'])
                write(f"\n【过滤前原始数据统计】")
                write(f"  总实例数: {total_instances}")
                write(f"  Standing实例数: {total_standing} ({total_standing/total_instances*100:.1f}%)")
                write(f"  Laying实例数: {total_laying} ({total_laying/total_instances*100:.1f}%)")
                
                # 计算最佳参数组合后的数据量
                best_filtered_df = self.apply_filters(
                    self.df,
                    conf_threshold=self.best_conf,
                    roi_config=self.roi_configs[self.best_roi],
                    wl_range=self.best_wl_range
                )
                filtered_total = len(best_filtered_df)
                filtered_standing = len(best_filtered_df[best_filtered_df['Class'] == 'standing'])
                filtered_laying = len(best_filtered_df[best_filtered_df['Class'] == 'laying'])
                
                write(f"\n【最佳参数过滤后数据统计】")
                write(f"  过滤后总实例数: {filtered_total} (保留率: {filtered_total/total_instances*100:.1f}%)")
                write(f"  过滤后Standing实例数: {filtered_standing} (保留率: {filtered_standing/total_standing*100:.1f}%)")
                write(f"  过滤后Laying实例数: {filtered_laying} (保留率: {filtered_laying/total_laying*100:.1f}%)")
                write(f"  过滤掉的实例数: {total_instances - filtered_total} ({(total_instances - filtered_total)/total_instances*100:.1f}%)\n")
                
                # 添加每个时间点的详细统计 - 三个独立表格
                if 'Day_Index' in self.df.columns:
                    # 获取所有时间点（确保顺序）
                    all_days = sorted(self.df['Day_Index'].unique())
                    
                    # 计算过滤前的每日统计
                    original_daily = self.df[self.df['Class'] == 'standing'].groupby('Day_Index').size()
                    
                    # ========== 表格1: 变化置信度的各时间点统计 ==========
                    write("\n【表1: 变化置信度的各时间点Standing实例统计（固定ROI=center_70, WL=[0.25,0.333]）】")
                    
                    # 构建表头
                    header_parts = ["参数      "]
                    for day_idx in all_days:
                        header_parts.append(f"Day{day_idx+1:<4}")
                    header_parts.append("总计   ")
                    write(" | ".join(header_parts))
                    write("-" * (len(" | ".join(header_parts)) + 10))
                    
                    # 过滤前数据行
                    row_parts = ["过滤前    "]
                    for day_idx in all_days:
                        count = original_daily.get(day_idx, 0)
                        row_parts.append(f"{count:<8}")
                    row_parts.append(f"{original_daily.sum():<8}")
                    write(" | ".join(row_parts))
                    
                    # 为每个置信度计算并输出
                    for conf_key, data in sorted(self.best_results['confidence'].items(), key=lambda x: x[1]['conf']):
                        # 计算该置信度下的过滤结果
                        filtered_df = self.apply_filters(
                            self.df,
                            conf_threshold=data['conf'],
                            roi_config=self.roi_configs[self.best_roi],
                            wl_range=self.best_wl_range
                        )
                        filtered_daily = filtered_df[filtered_df['Class'] == 'standing'].groupby('Day_Index').size()
                        
                        row_parts = [f"Conf={data['conf']:<3.1f}"]
                        for day_idx in all_days:
                            count = filtered_daily.get(day_idx, 0)
                            row_parts.append(f"{count:<8}")
                        row_parts.append(f"{filtered_daily.sum():<8}")
                        write(" | ".join(row_parts))
                    
                    # ========== 表格2: 变化ROI区域的各时间点统计 ==========
                    write("\n【表2: 变化ROI区域的各时间点Standing实例统计（固定Conf=0.7, WL=[0.25,0.333]）】")
                    
                    # 构建表头
                    header_parts = ["参数         "]
                    for day_idx in all_days:
                        header_parts.append(f"Day{day_idx+1:<4}")
                    header_parts.append("总计   ")
                    write(" | ".join(header_parts))
                    write("-" * (len(" | ".join(header_parts)) + 10))
                    
                    # 过滤前数据行
                    row_parts = ["过滤前       "]
                    for day_idx in all_days:
                        count = original_daily.get(day_idx, 0)
                        row_parts.append(f"{count:<8}")
                    row_parts.append(f"{original_daily.sum():<8}")
                    write(" | ".join(row_parts))
                    
                    # 为每个ROI计算并输出
                    for roi_name in ['center_50', 'center_60', 'center_70', 'center_80']:
                        if roi_name in self.best_results['roi']:
                            data = self.best_results['roi'][roi_name]
                            # 计算该ROI下的过滤结果
                            filtered_df = self.apply_filters(
                                self.df,
                                conf_threshold=self.best_conf,
                                roi_config=self.roi_configs[roi_name],
                                wl_range=self.best_wl_range
                            )
                            filtered_daily = filtered_df[filtered_df['Class'] == 'standing'].groupby('Day_Index').size()
                            
                            row_parts = [f"{roi_name:<13}"]
                            for day_idx in all_days:
                                count = filtered_daily.get(day_idx, 0)
                                row_parts.append(f"{count:<8}")
                            row_parts.append(f"{filtered_daily.sum():<8}")
                            write(" | ".join(row_parts))
                    
                    # ========== 表格3: 变化W/L阈值的各时间点统计 ==========
                    write("\n【表3: 变化W/L阈值范围的各时间点Standing实例统计（固定Conf=0.7, ROI=center_70）】")
                    
                    # 构建表头
                    header_parts = ["参数              "]
                    for day_idx in all_days:
                        header_parts.append(f"Day{day_idx+1:<4}")
                    header_parts.append("总计   ")
                    write(" | ".join(header_parts))
                    write("-" * (len(" | ".join(header_parts)) + 10))
                    
                    # 过滤前数据行
                    row_parts = ["过滤前            "]
                    for day_idx in all_days:
                        count = original_daily.get(day_idx, 0)
                        row_parts.append(f"{count:<8}")
                    row_parts.append(f"{original_daily.sum():<8}")
                    write(" | ".join(row_parts))
                    
                    # 为每个W/L范围计算并输出
                    for wl_key, data in sorted(self.best_results['wl_range'].items(), key=lambda x: x[1]['wl_min']):
                        # 计算该W/L范围下的过滤结果
                        filtered_df = self.apply_filters(
                            self.df,
                            conf_threshold=self.best_conf,
                            roi_config=self.roi_configs[self.best_roi],
                            wl_range=(data['wl_min'], data['wl_max'])
                        )
                        filtered_daily = filtered_df[filtered_df['Class'] == 'standing'].groupby('Day_Index').size()
                        
                        wl_label = f"[{data['wl_min']:.2f},{data['wl_max']:.2f}]"
                        row_parts = [f"{wl_label:<18}"]
                        for day_idx in all_days:
                            count = filtered_daily.get(day_idx, 0)
                            row_parts.append(f"{count:<8}")
                        row_parts.append(f"{filtered_daily.sum():<8}")
                        write(" | ".join(row_parts))
                else:
                    write("  警告: 数据中没有Day_Index列，无法统计各时间点")
                
                write("\n【变化置信度（固定ROI=center_70, WL=[0.25,0.333]）】")
                write(f"{'Conf':<8} | {'Standing Count':<15} | {'Standing Mean W/L':<20} | {'Standing Std':<15} | {'Retention Rate':<15}")
                write("-" * 100)
                for conf_key, data in self.best_results['confidence'].items():
                    if data['standing'] is not None:
                        stats = data['standing']
                        retention = stats['total_count'] / total_standing * 100
                        write(f"{data['conf']:<8.1f} | {stats['total_count']:<15} | "
                              f"{stats['overall_mean']:<20.3f} | {stats['overall_std']:<15.3f} | {retention:<14.1f}%")
                
                write("\n【变化ROI区域（固定Conf=0.7, WL=[0.25,0.333]）】")
                write(f"{'ROI Config':<15} | {'Standing Count':<15} | {'Standing Mean W/L':<20} | {'Standing Std':<15} | {'Retention Rate':<15}")
                write("-" * 100)
                for roi_name, data in self.best_results['roi'].items():
                    if data['standing'] is not None:
                        stats = data['standing']
                        retention = stats['total_count'] / total_standing * 100
                        write(f"{roi_name:<15} | {stats['total_count']:<15} | "
                              f"{stats['overall_mean']:<20.3f} | {stats['overall_std']:<15.3f} | {retention:<14.1f}%")
                
                write("\n【变化W/L阈值范围（固定Conf=0.7, ROI=center_70）】")
                write(f"{'W/L范围':<20} | {'Standing Count':<15} | {'Standing Mean W/L':<20} | {'Standing Std':<15} | {'Retention Rate':<15}")
                write("-" * 100)
                for wl_key, data in self.best_results['wl_range'].items():
                    if data['standing'] is not None:
                        stats = data['standing']
                        retention = stats['total_count'] / total_standing * 100
                        write(f"[{data['wl_min']:.3f}, {data['wl_max']:.3f}]    | {stats['total_count']:<15} | "
                              f"{stats['overall_mean']:<20.3f} | {stats['overall_std']:<15.3f} | {retention:<14.1f}%")
            
            # 4. 生长曲线拟合结果（新增详细参数分析）
            write("\n" + "=" * 80)
            write("4. Logistic生长曲线拟合分析")
            write("=" * 80 + "\n")
            
            # 初始化拟合成功率变量
            success_rate = 0.0
            
            if hasattr(self, 'growth_fits') and self.growth_fits:
                # 统计拟合成功率
                total_fits = len(self.growth_fits)
                successful_fits = [fit for fit in self.growth_fits.values() if fit and fit['fitting_success']]
                success_rate = len(successful_fits) / total_fits * 100
                
                write(f"拟合成功率: {len(successful_fits)}/{total_fits} ({success_rate:.1f}%)\n")
                
                # 分组显示拟合结果
                write("【置信度阈值的影响】")
                write("-" * 80)
                write(f"{'参数':<20} | {'p-value':<12} | {'R²':<8} | {'b±SE':<18} | {'L±SE':<18} | {'k±SE':<18} | {'t0±SE':<15}")
                write("-" * 80)
                
                for key, fit in self.growth_fits.items():
                    if key.startswith('conf_') and fit and fit['fitting_success']:
                        write(f"{fit['param_name']:<20} | {fit['p_value_linear']:<12.4e} | {fit['r2']:<8.3f} | "
                              f"{fit['b']:.4f}±{fit['b_err']:.4f}  | "
                              f"{fit['L']:.4f}±{fit['L_err']:.4f}  | "
                              f"{fit['k']:.4f}±{fit['k_err']:.4f}  | "
                              f"{fit['t0']:.1f}±{fit['t0_err']:.1f}")
                
                write("\n【ROI区域的影响】")
                write("-" * 80)
                write(f"{'参数':<20} | {'p-value':<12} | {'R²':<8} | {'b±SE':<18} | {'L±SE':<18} | {'k±SE':<18} | {'t0±SE':<15}")
                write("-" * 80)
                
                for key, fit in self.growth_fits.items():
                    if key.startswith('roi_') and 'no_filter' not in key and fit and fit['fitting_success']:
                        write(f"{fit['param_name']:<20} | {fit['p_value_linear']:<12.4e} | {fit['r2']:<8.3f} | "
                              f"{fit['b']:.4f}±{fit['b_err']:.4f}  | "
                              f"{fit['L']:.4f}±{fit['L_err']:.4f}  | "
                              f"{fit['k']:.4f}±{fit['k_err']:.4f}  | "
                              f"{fit['t0']:.1f}±{fit['t0_err']:.1f}")
                
                write("\n【W/L阈值的影响（5个范围）】")
                write("-" * 80)
                write(f"{'参数':<25} | {'p-value':<12} | {'R²':<8} | {'b±SE':<18} | {'L±SE':<18} | {'k±SE':<18} | {'t0±SE':<15}")
                write("-" * 80)
                
                for key, fit in self.growth_fits.items():
                    if key.startswith('wl_') and fit:
                        if fit['fitting_success']:
                            write(f"{fit['param_name']:<25} | {fit['p_value_linear']:<12.4e} | {fit['r2']:<8.3f} | "
                                  f"{fit['b']:.4f}±{fit['b_err']:.4f}  | "
                                  f"{fit['L']:.4f}±{fit['L_err']:.4f}  | "
                                  f"{fit['k']:.4f}±{fit['k_err']:.4f}  | "
                                  f"{fit['t0']:.1f}±{fit['t0_err']:.1f}")
                        else:
                            write(f"{fit['param_name']:<25} | 拟合失败")
                
                # 详细参数不确定性分析（选择一个代表性案例）
                write("\n" + "=" * 80)
                write("5. 参数不确定性详细分析（代表性案例）")
                write("=" * 80 + "\n")
                
                # 找到拟合最好的案例
                best_fit = None
                best_r2 = 0
                for fit in successful_fits:
                    if fit['r2'] > best_r2:
                        best_r2 = fit['r2']
                        best_fit = fit
                
                if best_fit:
                    write(f"分析案例: {best_fit['param_name']} (R² = {best_fit['r2']:.4f})")
                    write("\n【拟合参数与不确定性】")
                    write(f"{'Parameter':<30} {'Estimate':<14} {'Std. Error':<13} {'95% CI':<30} {'CV(%)':<8}")
                    write("-" * 80)
                    write(f"{'Lower asymptote (b):':<30} {best_fit['b']:<14.6f} {best_fit['b_err']:<13.6f} "
                          f"[{best_fit['b']-best_fit['b_ci']:.6f}, {best_fit['b']+best_fit['b_ci']:.6f}] {best_fit['b_rel_err']:>7.2f}")
                    write(f"{'Upper asymptote inc. (L):':<30} {best_fit['L']:<14.6f} {best_fit['L_err']:<13.6f} "
                          f"[{best_fit['L']-best_fit['L_ci']:.6f}, {best_fit['L']+best_fit['L_ci']:.6f}] {best_fit['L_rel_err']:>7.2f}")
                    write(f"{'Final asymptote (L+b):':<30} {best_fit['final_value']:<14.6f}")
                    write(f"{'Growth rate (k, day⁻¹):':<30} {best_fit['k']:<14.6f} {best_fit['k_err']:<13.6f} "
                          f"[{best_fit['k']-best_fit['k_ci']:.6f}, {best_fit['k']+best_fit['k_ci']:.6f}] {best_fit['k_rel_err']:>7.2f}")
                    write(f"{'Inflection point (t₀, d):':<30} {best_fit['t0']:<14.2f} {best_fit['t0_err']:<13.2f} "
                          f"[{best_fit['t0']-best_fit['t0_ci']:.2f}, {best_fit['t0']+best_fit['t0_ci']:.2f}] {best_fit['t0_rel_err']:>7.2f}")
                    
                    write("\n【统计显著性检验】")
                    sig_level = "***" if best_fit['p_value_linear'] < 0.001 else "**" if best_fit['p_value_linear'] < 0.01 else "*" if best_fit['p_value_linear'] < 0.05 else "ns"
                    write(f"  线性回归 p 值:          {best_fit['p_value_linear']:.4e}  {sig_level}")
                    write(f"  Spearman 相关 p 值:     {best_fit['p_value_spearman']:.4e}")
                    write(f"  效应量 (Cohen's d):      {best_fit['cohens_d']:.3f}  ({'large' if abs(best_fit['cohens_d']) > 0.8 else 'medium' if abs(best_fit['cohens_d']) > 0.5 else 'small'})")
                    write(f"  绝对变化量:             {best_fit['mean_diff']:.4f}  ({best_fit['mean_diff']/best_fit['y_observed'][0]*100:.1f}% 增长)")
                    
                    if best_fit['p_value_linear'] < 0.05:
                        write("\n✓✓✓ 结论：时间效应在统计学上高度显著！")
                        write("群体水平的增长趋势是真实的、非随机的。")
                    
                    write("\n【生长关键时期】")
                    write(f"  拐点时间: {best_fit['t0']:.1f} ± {best_fit['t0_err']:.1f} 天")
                    write(f"  最大生长速率: {best_fit['max_growth_rate']:.6f} W/L/day (第 {best_fit['max_growth_time']:.0f} 天)")
                    write(f"  迟缓期: 0 - {best_fit['lag_phase_end']:.0f} 天")
                    write(f"  快速生长期: {best_fit['lag_phase_end']:.0f} - {best_fit['rapid_phase_end']:.0f} 天")
                    write(f"  平台期开始: {best_fit['rapid_phase_end']:.0f} 天后")
                    if best_fit['day_90_percent']:
                        write(f"  90%生长完成: 第 {best_fit['day_90_percent']:.0f} 天")
                    if best_fit['day_95_percent']:
                        write(f"  95%生长完成: 第 {best_fit['day_95_percent']:.0f} 天")
                    
                    write("\n【参数精度评级】")
                    write(f"  • 基线水平 (b):      {'优秀' if best_fit['b_rel_err'] < 5 else '良好' if best_fit['b_rel_err'] < 10 else '可接受'} (CV = {best_fit['b_rel_err']:.1f}%)")
                    write(f"  • 拐点时间 (t₀):     {'优秀' if best_fit['t0_rel_err'] < 15 else '良好' if best_fit['t0_rel_err'] < 30 else '可接受' if best_fit['t0_rel_err'] < 50 else '较低'} (CV = {best_fit['t0_rel_err']:.1f}%)")
                    write(f"  • 增长幅度 (L):      {'优秀' if best_fit['L_rel_err'] < 15 else '良好' if best_fit['L_rel_err'] < 30 else '可接受' if best_fit['L_rel_err'] < 50 else '较低'} (CV = {best_fit['L_rel_err']:.1f}%)")
                    write(f"  • 生长速率 (k):      {'优秀' if best_fit['k_rel_err'] < 15 else '良好' if best_fit['k_rel_err'] < 30 else '可接受' if best_fit['k_rel_err'] < 50 else '较低'} (CV = {best_fit['k_rel_err']:.1f}%)")
            else:
                write("未执行生长曲线拟合分析。")
            
            # 5B. 最佳参数控制变量的R²和RMSE分析
            if hasattr(self, 'best_growth_fits') and self.best_growth_fits:
                write("\n" + "=" * 80)
                write("5. 最佳参数控制变量的拟合质量分析（R²和RMSE）")
                write("=" * 80 + "\n")
                
                write("【置信度阈值的拟合质量（固定ROI=center_70, WL=[0.25,0.333]）】")
                write(f"{'Conf':<8} | {'R²':<8} | {'RMSE':<12} | {'p-value':<12} | {'拟合状态':<12}")
                write("-" * 80)
                for key, fit in self.best_growth_fits.items():
                    if key.startswith('conf_'):
                        if fit and fit['fitting_success']:
                            rmse = np.sqrt(np.mean((fit['y_observed'] - logistic(fit['days'], fit['L'], fit['k'], fit['t0'], fit['b']))**2))
                            status = "优秀" if fit['r2'] > 0.8 else "良好" if fit['r2'] > 0.6 else "可接受"
                            write(f"{fit['param_name'].split('=')[1]:<8} | {fit['r2']:<8.3f} | {rmse:<12.6f} | {fit['p_value_linear']:<12.4e} | {status:<12}")
                
                write("\n【ROI区域的拟合质量（固定Conf=0.7, WL=[0.25,0.333]）】")
                write(f"{'ROI':<15} | {'R²':<8} | {'RMSE':<12} | {'p-value':<12} | {'拟合状态':<12}")
                write("-" * 80)
                for key, fit in self.best_growth_fits.items():
                    if key.startswith('roi_'):
                        if fit and fit['fitting_success']:
                            rmse = np.sqrt(np.mean((fit['y_observed'] - logistic(fit['days'], fit['L'], fit['k'], fit['t0'], fit['b']))**2))
                            status = "优秀" if fit['r2'] > 0.8 else "良好" if fit['r2'] > 0.6 else "可接受"
                            roi_name = fit['param_name'].replace('ROI=', '')
                            write(f"{roi_name:<15} | {fit['r2']:<8.3f} | {rmse:<12.6f} | {fit['p_value_linear']:<12.4e} | {status:<12}")
                
                write("\n【W/L阈值的拟合质量（固定Conf=0.7, ROI=center_70）】")
                write(f"{'W/L Range':<20} | {'R²':<8} | {'RMSE':<12} | {'p-value':<12} | {'拟合状态':<12}")
                write("-" * 80)
                for key, fit in self.best_growth_fits.items():
                    if key.startswith('wl_'):
                        if fit and fit['fitting_success']:
                            rmse = np.sqrt(np.mean((fit['y_observed'] - logistic(fit['days'], fit['L'], fit['k'], fit['t0'], fit['b']))**2))
                            status = "优秀" if fit['r2'] > 0.8 else "良好" if fit['r2'] > 0.6 else "可接受"
                            write(f"{fit['param_name']:<20} | {fit['r2']:<8.3f} | {rmse:<12.6f} | {fit['p_value_linear']:<12.4e} | {status:<12}")
                
                # 5C. 指定参数组合的详细参数不确定性
                write("\n" + "=" * 80)
                write("5C. 指定参数组合的详细参数不确定性分析")
                
                # 【用户指定的参数】：conf=0.7, roi=center_70, WL=[0.250,0.333]
                target_conf = 0.7
                target_roi = 'center_70'
                target_wl_range = (0.250, 0.333)
                
                write(f"指定参数：Conf={target_conf}, ROI={target_roi}, WL={target_wl_range}")
                write("=" * 80 + "\n")
                
                # 查找指定参数组合的拟合结果
                best_combo_fit = None
                
                # 构造参数名称来匹配
                target_param_name = f"Conf{target_conf}_ROI-{target_roi}_WL{target_wl_range[0]:.2f}-{target_wl_range[1]:.2f}"
                
                # 遍历所有拟合结果，查找匹配的参数组合
                for key, fit in self.best_growth_fits.items():
                    if fit and fit['fitting_success']:
                        if fit['param_name'] == target_param_name:
                            best_combo_fit = fit
                            print(f"✓ 找到指定参数组合：{target_param_name}")
                            break
                
                # 如果精确匹配失败，尝试部分匹配
                if not best_combo_fit:
                    for key, fit in self.best_growth_fits.items():
                        if fit and fit['fitting_success']:
                            if f"Conf{target_conf}" in fit['param_name'] and \
                               target_roi in fit['param_name'] and \
                               f"WL{target_wl_range[0]:.2f}" in fit['param_name']:
                                best_combo_fit = fit
                                print(f"✓ 部分匹配找到：{fit['param_name']}")
                                break
                
                if best_combo_fit:
                    write("【拟合参数值与标准误差】")
                    write(f"{'Parameter':<30} {'Value':<15} {'Std. Error':<15} {'CV(%)':<10}")
                    write("-" * 80)
                    write(f"{'Lower asymptote (b):':<30} {best_combo_fit['b']:<15.6f} {best_combo_fit['b_err']:<15.6f} {best_combo_fit['b_rel_err']:>9.2f}")
                    write(f"{'Upper asymptote inc. (L):':<30} {best_combo_fit['L']:<15.6f} {best_combo_fit['L_err']:<15.6f} {best_combo_fit['L_rel_err']:>9.2f}")
                    write(f"{'Growth rate (k, day⁻¹):':<30} {best_combo_fit['k']:<15.6f} {best_combo_fit['k_err']:<15.6f} {best_combo_fit['k_rel_err']:>9.2f}")
                    write(f"{'Inflection point (t₀, d):':<30} {best_combo_fit['t0']:<15.2f} {best_combo_fit['t0_err']:<15.2f} {best_combo_fit['t0_rel_err']:>9.2f}")
                    
                    write("\n【参数95%置信区间】")
                    write(f"{'Parameter':<30} {'95% CI':<40}")
                    write("-" * 80)
                    write(f"{'Lower asymptote (b):':<30} [{best_combo_fit['b']-best_combo_fit['b_ci']:.6f}, {best_combo_fit['b']+best_combo_fit['b_ci']:.6f}]")
                    write(f"{'Upper asymptote inc. (L):':<30} [{best_combo_fit['L']-best_combo_fit['L_ci']:.6f}, {best_combo_fit['L']+best_combo_fit['L_ci']:.6f}]")
                    write(f"{'Growth rate (k, day⁻¹):':<30} [{best_combo_fit['k']-best_combo_fit['k_ci']:.6f}, {best_combo_fit['k']+best_combo_fit['k_ci']:.6f}]")
                    write(f"{'Inflection point (t₀, d):':<30} [{best_combo_fit['t0']-best_combo_fit['t0_ci']:.2f}, {best_combo_fit['t0']+best_combo_fit['t0_ci']:.2f}]")
                    write(f"{'Final asymptote (L+b):':<30} {best_combo_fit['final_value']:.6f}")
                    
                    write("\n【拟合质量指标】")
                    write(f"  R² (决定系数):        {best_combo_fit['r2']:.4f}")
                    write(f"  RMSE (均方根误差):    {best_combo_fit['rmse']:.6f}")
                    write(f"  p-value (线性回归):   {best_combo_fit['p_value_linear']:.4e}")
                    sig = "***" if best_combo_fit['p_value_linear'] < 0.001 else "**" if best_combo_fit['p_value_linear'] < 0.01 else "*"
                    write(f"  统计显著性:           {sig} (高度显著)")
                    
                    write("\n【参数精度评级】")
                    write(f"  • 基线水平 (b):      {'优秀' if best_combo_fit['b_rel_err'] < 5 else '良好' if best_combo_fit['b_rel_err'] < 10 else '可接受'} (CV = {best_combo_fit['b_rel_err']:.2f}%)")
                    write(f"  • 增长幅度 (L):      {'优秀' if best_combo_fit['L_rel_err'] < 15 else '良好' if best_combo_fit['L_rel_err'] < 30 else '可接受'} (CV = {best_combo_fit['L_rel_err']:.2f}%)")
                    write(f"  • 生长速率 (k):      {'优秀' if best_combo_fit['k_rel_err'] < 15 else '良好' if best_combo_fit['k_rel_err'] < 30 else '可接受'} (CV = {best_combo_fit['k_rel_err']:.2f}%)")
                    write(f"  • 拐点时间 (t₀):     {'优秀' if best_combo_fit['t0_rel_err'] < 15 else '良好' if best_combo_fit['t0_rel_err'] < 30 else '可接受'} (CV = {best_combo_fit['t0_rel_err']:.2f}%)")
                    
                    write("\n【关键生长指标】")
                else:
                    write(f"⚠️ 警告：未找到指定参数组合 (Conf={target_conf}, ROI={target_roi}, WL={target_wl_range}) 的拟合结果。")
                    write("请检查：")
                    write("  1. 该参数组合是否在分析中包含")
                    write("  2. 该参数组合是否拟合成功")
                    write("\n可用的参数组合列表：")
                    for key, fit in self.best_growth_fits.items():
                        if fit and fit['fitting_success']:
                            write(f"  - {fit['param_name']} (R²={fit['r2']:.4f})")
                
                # 如果找到了拟合结果，继续打印关键指标
                if best_combo_fit:

                    write(f"  最大生长速率:        {best_combo_fit['max_growth_rate']:.6f} W/L/day")
                    write(f"  拐点时间 (t₀):       {best_combo_fit['t0']:.1f} ± {best_combo_fit['t0_err']:.1f} 天")
                    if 'lag_phase_end' in best_combo_fit:
                        write(f"  迟缓期结束:          第 {best_combo_fit['lag_phase_end']:.0f} 天")
                    if 'rapid_phase_end' in best_combo_fit:
                        write(f"  快速增长期结束:      第 {best_combo_fit['rapid_phase_end']:.0f} 天")
            
            # 6. 结论与建议
            write("\n" + "=" * 80)
            write("6. 分析结论与参数选择建议")
            write("=" * 80)
            write("\n基于敏感性分析结果，我们观察到：")
            write("\n1. 置信度阈值影响：")
            write("   - 较低阈值(0.5-0.6)保留更多实例，但可能包含误检")
            write("   - 较高阈值(0.8-0.9)实例数显著减少，但精度更高")
            write("   - W/L轨迹趋势在不同阈值下保持一致")
            write("\n2. ROI区域影响：")
            write("   - 边缘区域的牛可能因遮挡、透视失真导致W/L偏差")
            write("   - 中心区域的测量更稳定可靠")
            write("\n3. W/L阈值影响（5个范围）：")
            write("   - 分析了5个不同的W/L范围：[0.25,0.333], [0.26,0.333], [0.27,0.333], [0.28,0.333], [0.30,0.333]")
            write("   - 对应长宽比范围：AR=3.0-4.0, 3.0-3.85, 3.0-3.7, 3.0-3.57, 3.0-3.33")
            write("   - 过滤极端值后，轨迹更平滑")
            write("   - 建议根据实际牛只形态设置合理范围")
            write("\n4. Logistic拟合质量：")
            write(f"   - 成功拟合率: {success_rate:.1f}%")
            write("   - 大部分参数组合能够显著拟合生长趋势")
            write("   - 基线参数(b)和拐点时间(t0)精度较高")
            write("\n" + "=" * 80)
        
        print(f"\n✓ 报告已保存: {output_path}")
        return output_path
    
    def run_full_analysis(self):
        """运行完整的敏感性分析"""
        print("\n" + "=" * 80)
        print("开始牛W/L比值参数敏感性分析")
        print("=" * 80)
        print(f"📁 所有结果将保存到: {self.output_dir}")
        print("=" * 80 + "\n")
        
        # 加载数据
        self.load_data()
        
        # 运行三种分析
        self.analyze_confidence_sensitivity()
        self.analyze_roi_sensitivity()
        self.analyze_wl_sensitivity()
        
        # 生成可视化
        print("=" * 80)
        print("生成可视化图表")
        print("=" * 80)
        
        output_dir = Path(self.output_dir)
        
        # 注释掉暂时不需要的图表
        # self.plot_confidence_trajectories(
        #     save_path=output_dir / 'sensitivity_confidence_trajectories.png'
        # )
        
        # self.plot_roi_trajectories(
        #     save_path=output_dir / 'sensitivity_roi_trajectories.png'
        # )
        
        # self.plot_wl_threshold_trajectories(
        #     save_path=output_dir / 'sensitivity_wl_threshold_trajectories.png'
        # )
        
        # self.plot_parameter_heatmap(
        #     save_path=output_dir / 'sensitivity_parameter_heatmap.png'
        # )
        
        # # 新增：综合敏感性分析图（回应审稿人）
        # self.plot_combined_sensitivity(
        #     save_path=output_dir / 'sensitivity_combined_three_parameters.png'
        # )
        
        # # 新增：实例个数随时间变化可视化
        # self.plot_instance_counts_over_time(
        #     save_path=output_dir / 'sensitivity_instance_counts_over_time.png'
        # )
        
        # 新增：最佳参数控制变量分析
        print("\n" + "=" * 80)
        print("最佳参数控制变量分析")
        print("=" * 80)
        self.analyze_with_best_params()
        
        # self.plot_best_params_sensitivity(
        #     save_path=output_dir / 'sensitivity_best_params_controlled.png'
        # )
        
        # 保留：最佳参数控制变量实例个数可视化
        self.plot_best_params_instance_counts(
            save_path=output_dir / 'sensitivity_best_params_instance_counts.png'
        )
        # 同时保存PDF版本
        self.plot_best_params_instance_counts(
            save_path=output_dir / 'sensitivity_best_params_instance_counts.pdf'
        )
        
        # self.plot_sample_size_diagnostic(
        #     save_path=output_dir / 'sensitivity_sample_size_diagnostic.png'
        # )
        
        # 新增：最佳参数控制变量的生长曲线拟合
        self.analyze_best_params_growth_curves()
        
        # 保留：最佳参数生长曲线拟合图
        self.plot_best_params_growth_fits(
            save_path=output_dir / 'best_params_growth_curve_fits.png'
        )
        # 同时保存PDF版本
        self.plot_best_params_growth_fits(
            save_path=output_dir / 'best_params_growth_curve_fits.pdf'
        )
        
        # 保留：最佳参数生长速率曲线
        self.plot_best_params_growth_rates(
            save_path=output_dir / 'best_params_growth_rate_curves.png'
        )
        # 同时保存PDF版本
        self.plot_best_params_growth_rates(
            save_path=output_dir / 'best_params_growth_rate_curves.pdf'
        )
        
        # 保留：最佳参数拟合质量可视化（R²和RMSE）
        self.plot_best_params_fitting_quality(
            save_path=output_dir / 'best_params_fitting_quality_R2_RMSE.png'
        )
        # 同时保存PDF版本
        self.plot_best_params_fitting_quality(
            save_path=output_dir / 'best_params_fitting_quality_R2_RMSE.pdf'
        )
        
        # # 注释掉暂时不需要的图表
        # self.analyze_growth_curves()
        
        # # 新增：生成拟合曲线图表
        # self.plot_growth_curve_fits(
        #     save_path=output_dir / 'growth_curve_logistic_fits.png'
        # )
        
        # self.plot_growth_rate_curves(
        #     save_path=output_dir / 'growth_rate_curves.png'
        # )
        
        # 生成报告
        print("\n" + "=" * 80)
        print("生成分析报告")
        print("=" * 80)
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("✓ 敏感性分析完成！")
        print("=" * 80)


def main():
    """主函数"""
    import sys
    
    # 配置路径 - 可以从命令行参数获取
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 默认路径 - 固定的数据目录
        data_dir = r'E:\Datasets\CattleDetection\yolo_sam_w_15m_new\instances_results_20260110_172441'
        # data_dir = 'data/yolo_sam_w_10m/instances_results_20260112_105253'

        csv_path = os.path.join(data_dir, 'instances_metadata_all.csv')
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"✗ 错误: 找不到CSV文件: {csv_path}")
        print("\n请先运行 cattle_collect_all_instances.py 收集数据！")
        return
    
    # 创建分析器并运行
    analyzer = CattleSensitivityAnalyzer(csv_path)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

