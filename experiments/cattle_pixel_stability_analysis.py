"""
牛实例像素尺度与W/L稳定性分析
========================================

目的：验证猜想
1. 10m高度的牛像素更大但更不稳定（像素敏感）
2. 15m高度的牛像素更小但更稳定
3. W/L应该用 MinRect_Width/Head_Tail_Distance 而非 MinRect_Width/MinRect_Length

验证内容：
- 像素尺度统计对比（10m vs 15m）
- 变异系数(CV)分析
- 两种W/L定义的稳定性对比
- 相关性分析（像素尺度 vs W/L稳定性）
- 高度效应分解
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")


class PixelStabilityAnalyzer:
    """像素尺度与W/L稳定性分析器"""
    
    def __init__(self, csv_10m: str, csv_15m: str, output_dir: str = None):
        """
        初始化分析器
        
        Args:
            csv_10m: 10m高度数据CSV路径
            csv_15m: 15m高度数据CSV路径
            output_dir: 输出目录
        """
        self.csv_10m = csv_10m
        self.csv_15m = csv_15m
        self.df_10m = None
        self.df_15m = None
        self.df_combined = None
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path(f'./pixel_stability_analysis_{timestamp}')
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
        
        self.analysis_results = {}
        
    def load_data(self):
        """加载并预处理数据"""
        print("\n" + "=" * 80)
        print("加载数据...")
        print("=" * 80)
        
        # 加载10m数据
        print(f"📄 加载10m数据: {self.csv_10m}")
        self.df_10m_raw = pd.read_csv(self.csv_10m)
        self.df_10m_raw['Height'] = '10m'
        print(f"  ✓ 10m原始数据: {len(self.df_10m_raw)} 条记录")
        
        # 加载15m数据
        print(f"📄 加载15m数据: {self.csv_15m}")
        self.df_15m_raw = pd.read_csv(self.csv_15m)
        self.df_15m_raw['Height'] = '15m'
        print(f"  ✓ 15m原始数据: {len(self.df_15m_raw)} 条记录")
        
        # 计算新的W/L比值（MinRect_Width / Head_Tail_Distance）
        self.df_10m_raw['WL_New'] = self.df_10m_raw['MinRect_Width_pixels'] / self.df_10m_raw['Head_Tail_Distance_pixels']
        self.df_15m_raw['WL_New'] = self.df_15m_raw['MinRect_Width_pixels'] / self.df_15m_raw['Head_Tail_Distance_pixels']
        
        # 应用筛选条件
        print("\n" + "=" * 80)
        print("应用数据筛选条件...")
        print("=" * 80)
        self.apply_filters()
        
        # 合并数据
        self.df_combined = pd.concat([self.df_10m, self.df_15m], ignore_index=True)
        print(f"\n✅ 合并完成: 总共 {len(self.df_combined)} 条记录")
        print()
        
    def apply_filters(self):
        """应用严格的筛选条件"""
        print("\n筛选条件：")
        print("  1. Class == 'standing'")
        print("  2. ROI == 'center_70' (中心70%区域，归一化坐标: 0.15-0.85)")
        print("  3. W/L范围: 0.25 <= WL_Ratio <= 0.33")
        print("  4. MinRect_Length范围: 10m (790-950), 15m (490-600)")
        print()
        
        # ROI配置：center_70表示中心70%区域
        roi_x1, roi_y1, roi_x2, roi_y2 = 0.15, 0.15, 0.85, 0.85
        
        # W/L范围
        wl_min, wl_max = 0.25, 0.33
        
        # MinRect_Length范围
        length_10m_min, length_10m_max = 790, 950
        length_15m_min, length_15m_max = 490, 600
        
        # 统计过滤前后的数据
        filter_stats = []
        
        # 10m数据筛选
        print("📊 10m数据筛选:")
        df_10m_before = self.df_10m_raw.copy()
        print(f"  原始数据: {len(df_10m_before)} 条")
        
        # 按日期统计原始数据
        date_counts_before_10m = df_10m_before.groupby('Date').size().to_dict()
        
        # 条件1: standing
        df_10m_step1 = df_10m_before[df_10m_before['Class'] == 'standing']
        print(f"  ├─ Class='standing': {len(df_10m_step1)} 条 (保留 {len(df_10m_step1)/len(df_10m_before)*100:.1f}%)")
        
        # 条件2: center_70 ROI（基于BBox中心点归一化坐标）
        center_x_norm = df_10m_step1['BBox_Center_X'] / df_10m_step1['Image_Width']
        center_y_norm = df_10m_step1['BBox_Center_Y'] / df_10m_step1['Image_Height']
        roi_mask = (
            (center_x_norm >= roi_x1) & (center_x_norm <= roi_x2) &
            (center_y_norm >= roi_y1) & (center_y_norm <= roi_y2)
        )
        df_10m_step2 = df_10m_step1[roi_mask]
        print(f"  ├─ ROI='center_70': {len(df_10m_step2)} 条 (保留 {len(df_10m_step2)/len(df_10m_before)*100:.1f}%)")
        
        # 条件3: W/L范围
        df_10m_step3 = df_10m_step2[
            (df_10m_step2['WL_Ratio'] >= wl_min) &
            (df_10m_step2['WL_Ratio'] <= wl_max)
        ]
        print(f"  ├─ {wl_min} <= W/L <= {wl_max}: {len(df_10m_step3)} 条 (保留 {len(df_10m_step3)/len(df_10m_before)*100:.1f}%)")
        
        # 条件4: MinRect_Length范围
        df_10m_step4 = df_10m_step3[
            (df_10m_step3['MinRect_Length_pixels'] >= length_10m_min) &
            (df_10m_step3['MinRect_Length_pixels'] <= length_10m_max)
        ]
        print(f"  └─ {length_10m_min} <= Length <= {length_10m_max}: {len(df_10m_step4)} 条 (保留 {len(df_10m_step4)/len(df_10m_before)*100:.1f}%)")
        
        self.df_10m = df_10m_step4.copy()
        
        # 按日期统计过滤后数据
        date_counts_after_10m = self.df_10m.groupby('Date').size().to_dict()
        
        # 保存10m统计
        for date in sorted(set(list(date_counts_before_10m.keys()) + list(date_counts_after_10m.keys()))):
            before = date_counts_before_10m.get(date, 0)
            after = date_counts_after_10m.get(date, 0)
            filter_stats.append({
                'Height': '10m',
                'Date': date,
                'Before': before,
                'After': after,
                'Filtered': before - after,
                'Retention': after / before * 100 if before > 0 else 0
            })
        
        print(f"\n  总结: {len(df_10m_before)} → {len(self.df_10m)} 条 (过滤掉 {len(df_10m_before)-len(self.df_10m)} 条, 保留率 {len(self.df_10m)/len(df_10m_before)*100:.1f}%)")
        
        # 15m数据筛选
        print("\n📊 15m数据筛选:")
        df_15m_before = self.df_15m_raw.copy()
        print(f"  原始数据: {len(df_15m_before)} 条")
        
        # 按日期统计原始数据
        date_counts_before_15m = df_15m_before.groupby('Date').size().to_dict()
        
        # 条件1: standing
        df_15m_step1 = df_15m_before[df_15m_before['Class'] == 'standing']
        print(f"  ├─ Class='standing': {len(df_15m_step1)} 条 (保留 {len(df_15m_step1)/len(df_15m_before)*100:.1f}%)")
        
        # 条件2: center_70 ROI（基于BBox中心点归一化坐标）
        center_x_norm = df_15m_step1['BBox_Center_X'] / df_15m_step1['Image_Width']
        center_y_norm = df_15m_step1['BBox_Center_Y'] / df_15m_step1['Image_Height']
        roi_mask = (
            (center_x_norm >= roi_x1) & (center_x_norm <= roi_x2) &
            (center_y_norm >= roi_y1) & (center_y_norm <= roi_y2)
        )
        df_15m_step2 = df_15m_step1[roi_mask]
        print(f"  ├─ ROI='center_70': {len(df_15m_step2)} 条 (保留 {len(df_15m_step2)/len(df_15m_before)*100:.1f}%)")
        
        # 条件3: W/L范围
        df_15m_step3 = df_15m_step2[
            (df_15m_step2['WL_Ratio'] >= wl_min) &
            (df_15m_step2['WL_Ratio'] <= wl_max)
        ]
        print(f"  ├─ {wl_min} <= W/L <= {wl_max}: {len(df_15m_step3)} 条 (保留 {len(df_15m_step3)/len(df_15m_before)*100:.1f}%)")
        
        # 条件4: MinRect_Length范围
        df_15m_step4 = df_15m_step3[
            (df_15m_step3['MinRect_Length_pixels'] >= length_15m_min) &
            (df_15m_step3['MinRect_Length_pixels'] <= length_15m_max)
        ]
        print(f"  └─ {length_15m_min} <= Length <= {length_15m_max}: {len(df_15m_step4)} 条 (保留 {len(df_15m_step4)/len(df_15m_before)*100:.1f}%)")
        
        self.df_15m = df_15m_step4.copy()
        
        # 按日期统计过滤后数据
        date_counts_after_15m = self.df_15m.groupby('Date').size().to_dict()
        
        # 保存15m统计
        for date in sorted(set(list(date_counts_before_15m.keys()) + list(date_counts_after_15m.keys()))):
            before = date_counts_before_15m.get(date, 0)
            after = date_counts_after_15m.get(date, 0)
            filter_stats.append({
                'Height': '15m',
                'Date': date,
                'Before': before,
                'After': after,
                'Filtered': before - after,
                'Retention': after / before * 100 if before > 0 else 0
            })
        
        print(f"\n  总结: {len(df_15m_before)} → {len(self.df_15m)} 条 (过滤掉 {len(df_15m_before)-len(self.df_15m)} 条, 保留率 {len(self.df_15m)/len(df_15m_before)*100:.1f}%)")
        
        # 保存过滤统计
        self.filter_stats_df = pd.DataFrame(filter_stats)
        
        # 总体统计
        total_before = len(df_10m_before) + len(df_15m_before)
        total_after = len(self.df_10m) + len(self.df_15m)
        print("\n" + "=" * 80)
        print(f"总体: {total_before} → {total_after} 条 (过滤掉 {total_before-total_after} 条, 保留率 {total_after/total_before*100:.1f}%)")
        print("=" * 80)
        
    def analyze_pixel_scale(self):
        """分析像素尺度统计"""
        print("=" * 80)
        print("分析1: 像素尺度统计对比（10m vs 15m）")
        print("=" * 80)
        
        features = [
            'Area_pixels',
            'MinRect_Length_pixels',
            'MinRect_Width_pixels',
            'Head_Tail_Distance_pixels',
            'BBox_Width_pixels',
            'BBox_Height_pixels'
        ]
        
        results = {}
        
        for feature in features:
            print(f"\n📊 {feature}:")
            
            data_10m = self.df_10m[feature].dropna()
            data_15m = self.df_15m[feature].dropna()
            
            # 基本统计
            stats_10m = {
                'mean': data_10m.mean(),
                'std': data_10m.std(),
                'cv': data_10m.std() / data_10m.mean() * 100,  # 变异系数
                'median': data_10m.median(),
                'q25': data_10m.quantile(0.25),
                'q75': data_10m.quantile(0.75)
            }
            
            stats_15m = {
                'mean': data_15m.mean(),
                'std': data_15m.std(),
                'cv': data_15m.std() / data_15m.mean() * 100,
                'median': data_15m.median(),
                'q25': data_15m.quantile(0.25),
                'q75': data_15m.quantile(0.75)
            }
            
            # t检验
            t_stat, p_value = stats.ttest_ind(data_10m, data_15m)
            
            # 输出
            print(f"  10m: 均值={stats_10m['mean']:.1f}, 标准差={stats_10m['std']:.1f}, CV={stats_10m['cv']:.2f}%")
            print(f"  15m: 均值={stats_15m['mean']:.1f}, 标准差={stats_15m['std']:.1f}, CV={stats_15m['cv']:.2f}%")
            print(f"  均值比值(10m/15m): {stats_10m['mean']/stats_15m['mean']:.3f}")
            print(f"  CV比值(10m/15m): {stats_10m['cv']/stats_15m['cv']:.3f}")
            print(f"  t检验: t={t_stat:.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
            
            results[feature] = {
                '10m': stats_10m,
                '15m': stats_15m,
                't_stat': t_stat,
                'p_value': p_value
            }
        
        self.analysis_results['pixel_scale'] = results
        print()
        
    def analyze_wl_comparison(self):
        """对比两种W/L定义"""
        print("=" * 80)
        print("分析2: 两种W/L定义的对比")
        print("=" * 80)
        
        # 数据已经过滤为standing + center_70 + MinRect_Length范围
        df_10m_standing = self.df_10m
        df_15m_standing = self.df_15m
        
        wl_metrics = []
        
        for height, df in [('10m', df_10m_standing), ('15m', df_15m_standing)]:
            print(f"\n📊 {height} 高度:")
            
            # 原始W/L (MinRect_Width / MinRect_Length)
            wl_original = df['WL_Ratio'].dropna()
            
            # 新W/L (MinRect_Width / Head_Tail_Distance)
            wl_new = df['WL_New'].dropna()
            
            # 统计
            for wl_type, wl_data in [('Original (W/L)', wl_original), ('New (W/HTD)', wl_new)]:
                mean_val = wl_data.mean()
                std_val = wl_data.std()
                cv_val = std_val / mean_val * 100
                
                print(f"  {wl_type:20s}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, CV={cv_val:.2f}%")
                
                wl_metrics.append({
                    'Height': height,
                    'WL_Type': wl_type,
                    'Mean': mean_val,
                    'Std': std_val,
                    'CV': cv_val
                })
        
        self.analysis_results['wl_comparison'] = pd.DataFrame(wl_metrics)
        print()
        
    def analyze_wl_temporal_stability(self):
        """分析W/L的时间序列稳定性 - 关注时间点之间的变化规律"""
        print("=" * 80)
        print("分析3: 时间序列变化规律性分析（每个时间点10m vs 15m）")
        print("=" * 80)
        
        results = {}
        
        for height, df in [('10m', self.df_10m), ('15m', self.df_15m)]:
            df_standing = df  # 数据已过滤
            
            # 按日期和Day_Index分组
            grouped = df_standing.groupby(['Date', 'Day_Index']).agg({
                'WL_Ratio': ['mean', 'std', 'count'],
                'WL_New': ['mean', 'std', 'count'],
                'MinRect_Width_pixels': ['mean', 'std'],
                'MinRect_Length_pixels': ['mean', 'std']
            }).reset_index()
            
            # 按时间排序
            grouped = grouped.sort_values('Day_Index')
            
            # 提取均值序列
            wl_original_means = grouped[('WL_Ratio', 'mean')].values
            wl_new_means = grouped[('WL_New', 'mean')].values
            width_means = grouped[('MinRect_Width_pixels', 'mean')].values
            length_means = grouped[('MinRect_Length_pixels', 'mean')].values
            days = grouped['Day_Index'].values
            
            # 计算时间点之间的变化（一阶差分）
            wl_original_diff = np.diff(wl_original_means)
            wl_new_diff = np.diff(wl_new_means)
            width_diff = np.diff(width_means)
            length_diff = np.diff(length_means)
            
            # 计算变化的绝对值和相对值
            wl_original_abs_change = np.abs(wl_original_diff)
            wl_new_abs_change = np.abs(wl_new_diff)
            
            # 变化率（相对于均值的百分比）
            wl_original_pct_change = np.abs(wl_original_diff / wl_original_means[:-1] * 100)
            wl_new_pct_change = np.abs(wl_new_diff / wl_new_means[:-1] * 100)
            
            # 统计指标
            print(f"\n📊 {height} 时间序列分析:")
            print(f"  时间点数量: {len(days)}")
            print(f"\n  Original W/L (MinRect_Width/MinRect_Length):")
            print(f"    均值范围: {wl_original_means.min():.4f} - {wl_original_means.max():.4f}")
            print(f"    时间序列CV: {wl_original_means.std() / wl_original_means.mean() * 100:.3f}%")
            print(f"    相邻点平均变化: {wl_original_abs_change.mean():.4f}")
            print(f"    相邻点平均变化率: {wl_original_pct_change.mean():.3f}%")
            print(f"    变化率标准差: {wl_original_pct_change.std():.3f}%  (越小越规律)")
            print(f"    最大单次变化: {wl_original_abs_change.max():.4f} ({wl_original_pct_change.max():.2f}%)")
            
            print(f"\n  New W/L (MinRect_Width/Head_Tail_Distance):")
            print(f"    均值范围: {wl_new_means.min():.4f} - {wl_new_means.max():.4f}")
            print(f"    时间序列CV: {wl_new_means.std() / wl_new_means.mean() * 100:.3f}%")
            print(f"    相邻点平均变化: {wl_new_abs_change.mean():.4f}")
            print(f"    相邻点平均变化率: {wl_new_pct_change.mean():.3f}%")
            print(f"    变化率标准差: {wl_new_pct_change.std():.3f}%  (越小越规律)")
            print(f"    最大单次变化: {wl_new_abs_change.max():.4f} ({wl_new_pct_change.max():.2f}%)")
            
            print(f"\n  Width变化:")
            print(f"    相邻点平均变化: {np.abs(width_diff).mean():.2f} pixels")
            print(f"    变化率: {np.abs(width_diff / width_means[:-1] * 100).mean():.3f}%")
            
            print(f"\n  Length变化:")
            print(f"    相邻点平均变化: {np.abs(length_diff).mean():.2f} pixels")
            print(f"    变化率: {np.abs(length_diff / length_means[:-1] * 100).mean():.3f}%")
            
            results[height] = {
                'grouped_data': grouped,
                'days': days,
                'wl_original_means': wl_original_means,
                'wl_new_means': wl_new_means,
                'width_means': width_means,
                'length_means': length_means,
                'wl_original_diff': wl_original_diff,
                'wl_new_diff': wl_new_diff,
                'wl_original_abs_change': wl_original_abs_change,
                'wl_new_abs_change': wl_new_abs_change,
                'wl_original_pct_change': wl_original_pct_change,
                'wl_new_pct_change': wl_new_pct_change,
                'stats': {
                    'original_mean_change': wl_original_abs_change.mean(),
                    'original_pct_change': wl_original_pct_change.mean(),
                    'original_change_std': wl_original_pct_change.std(),
                    'new_mean_change': wl_new_abs_change.mean(),
                    'new_pct_change': wl_new_pct_change.mean(),
                    'new_change_std': wl_new_pct_change.std()
                }
            }
        
        # 对比10m vs 15m
        print("\n" + "=" * 80)
        print("10m vs 15m 对比:")
        print("=" * 80)
        
        stats_10m = results['10m']['stats']
        stats_15m = results['15m']['stats']
        
        print("\nOriginal W/L:")
        print(f"  10m 相邻点平均变化率: {stats_10m['original_pct_change']:.3f}%")
        print(f"  15m 相邻点平均变化率: {stats_15m['original_pct_change']:.3f}%")
        print(f"  {'10m更稳定' if stats_10m['original_pct_change'] < stats_15m['original_pct_change'] else '15m更稳定'}")
        
        print(f"\n  10m 变化率标准差: {stats_10m['original_change_std']:.3f}%  (规律性)")
        print(f"  15m 变化率标准差: {stats_15m['original_change_std']:.3f}%  (规律性)")
        print(f"  {'10m变化更规律' if stats_10m['original_change_std'] < stats_15m['original_change_std'] else '15m变化更规律'}")
        
        print("\nNew W/L:")
        print(f"  10m 相邻点平均变化率: {stats_10m['new_pct_change']:.3f}%")
        print(f"  15m 相邻点平均变化率: {stats_15m['new_pct_change']:.3f}%")
        print(f"  {'10m更稳定' if stats_10m['new_pct_change'] < stats_15m['new_pct_change'] else '15m更稳定'}")
        
        print(f"\n  10m 变化率标准差: {stats_10m['new_change_std']:.3f}%  (规律性)")
        print(f"  15m 变化率标准差: {stats_15m['new_change_std']:.3f}%  (规律性)")
        print(f"  {'10m变化更规律' if stats_10m['new_change_std'] < stats_15m['new_change_std'] else '15m变化更规律'}")
        
        self.analysis_results['temporal_stability'] = results
        print()
        
    def analyze_scale_vs_stability(self):
        """分析像素尺度与W/L稳定性的关系"""
        print("=" * 80)
        print("分析4: 像素尺度 vs W/L稳定性")
        print("=" * 80)
        
        # 数据已过滤
        df_standing = self.df_combined.copy()
        
        # 按Area分组（小、中、大）
        df_standing['Size_Group'] = pd.qcut(df_standing['Area_pixels'], 
                                             q=3, 
                                             labels=['Small', 'Medium', 'Large'])
        
        results = {}
        
        for height in ['10m', '15m']:
            df_height = df_standing[df_standing['Height'] == height]
            
            print(f"\n📊 {height}:")
            
            for size_group in ['Small', 'Medium', 'Large']:
                df_group = df_height[df_height['Size_Group'] == size_group]
                
                if len(df_group) > 10:
                    wl_original = df_group['WL_Ratio'].dropna()
                    wl_new = df_group['WL_New'].dropna()
                    
                    cv_original = wl_original.std() / wl_original.mean() * 100
                    cv_new = wl_new.std() / wl_new.mean() * 100
                    
                    print(f"  {size_group:8s}: n={len(df_group):4d}, "
                          f"CV_original={cv_original:.2f}%, CV_new={cv_new:.2f}%")
                    
                    if height not in results:
                        results[height] = {}
                    results[height][size_group] = {
                        'n': len(df_group),
                        'cv_original': cv_original,
                        'cv_new': cv_new
                    }
        
        self.analysis_results['scale_stability'] = results
        print()
        
    def analyze_correlation(self):
        """相关性分析"""
        print("=" * 80)
        print("分析5: 相关性分析")
        print("=" * 80)
        
        df_standing = self.df_combined.copy()  # 数据已过滤
        
        # 按日期和高度分组，计算每组的CV
        grouped = df_standing.groupby(['Date', 'Height']).agg({
            'Area_pixels': 'mean',
            'WL_Ratio': lambda x: x.std() / x.mean() * 100 if len(x) > 1 else np.nan,
            'WL_New': lambda x: x.std() / x.mean() * 100 if len(x) > 1 else np.nan
        }).reset_index()
        
        grouped.columns = ['Date', 'Height', 'Mean_Area', 'CV_WL_Original', 'CV_WL_New']
        grouped = grouped.dropna()
        
        print("\n相关性分析（面积 vs W/L变异性）:")
        
        for height in ['10m', '15m']:
            df_height = grouped[grouped['Height'] == height]
            
            if len(df_height) > 5:
                # 面积 vs 原始W/L的CV
                corr_original, p_original = pearsonr(df_height['Mean_Area'], 
                                                      df_height['CV_WL_Original'])
                
                # 面积 vs 新W/L的CV
                corr_new, p_new = pearsonr(df_height['Mean_Area'], 
                                            df_height['CV_WL_New'])
                
                print(f"\n  {height}:")
                print(f"    面积 vs CV_Original: r={corr_original:.3f}, p={p_original:.4f}")
                print(f"    面积 vs CV_New:      r={corr_new:.3f}, p={p_new:.4f}")
        
        self.analysis_results['correlation'] = grouped
        print()
        
    def analyze_normalization_necessity(self):
        """分析是否需要归一化：对比绝对值和相对增长率"""
        print("\n" + "=" * 80)
        print("分析6: 归一化必要性分析")
        print("=" * 80)
        
        features_to_analyze = [
            ('Area_pixels', 'Area'),
            ('MinRect_Length_pixels', 'Length'),
            ('MinRect_Width_pixels', 'Width'),
            ('Head_Tail_Distance_pixels', 'Head-Tail Distance'),
            ('Perimeter_pixels', 'Perimeter')
        ]
        
        normalization_results = {}
        
        for feature_col, feature_name in features_to_analyze:
            print(f"\n{'='*80}")
            print(f"特征: {feature_name}")
            print(f"{'='*80}")
            
            # 对10m和15m分别计算时序统计
            result_10m = self._calculate_temporal_stats(self.df_10m, feature_col, '10m')
            result_15m = self._calculate_temporal_stats(self.df_15m, feature_col, '15m')
            
            # 比较绝对值的差异
            print(f"\n1️⃣ 绝对值对比 (未归一化):")
            print(f"   10m均值: {result_10m['mean_value']:.1f} pixels")
            print(f"   15m均值: {result_15m['mean_value']:.1f} pixels")
            ratio = result_10m['mean_value'] / result_15m['mean_value']
            print(f"   比例(10m/15m): {ratio:.3f}x")
            print(f"   ⚠️  结论: 10m和15m的绝对值差异 {(ratio-1)*100:.1f}%")
            print(f"       → 无法直接比较！高度差异导致像素尺度完全不同")
            
            # 比较相对增长率（归一化到第一个时间点）
            print(f"\n2️⃣ 相对增长率对比 (归一化到首次测量):")
            print(f"   10m总增长率: {result_10m['total_growth']:.2f}%")
            print(f"   15m总增长率: {result_15m['total_growth']:.2f}%")
            growth_diff = abs(result_10m['total_growth'] - result_15m['total_growth'])
            print(f"   增长率差异: {growth_diff:.2f}%")
            print(f"   ✓ 结论: 归一化后可以直接比较生长趋势")
            
            # 比较时序变化的稳定性（CV）
            print(f"\n3️⃣ 时序变化稳定性:")
            print(f"   10m时序CV: {result_10m['temporal_cv']:.2f}%")
            print(f"   15m时序CV: {result_15m['temporal_cv']:.2f}%")
            more_stable = '10m' if result_10m['temporal_cv'] < result_15m['temporal_cv'] else '15m'
            print(f"   更稳定: {more_stable}")
            
            # 比较归一化后的变化率稳定性
            print(f"\n4️⃣ 归一化增长率的稳定性:")
            print(f"   10m相邻点变化率: {result_10m['avg_change_rate']:.3f}% (std: {result_10m['change_rate_std']:.3f}%)")
            print(f"   15m相邻点变化率: {result_15m['avg_change_rate']:.3f}% (std: {result_15m['change_rate_std']:.3f}%)")
            
            normalization_results[feature_name] = {
                '10m': result_10m,
                '15m': result_15m,
                'ratio': ratio,
                'growth_diff': growth_diff
            }
        
        # 总结
        print(f"\n{'='*80}")
        print("🎯 归一化必要性总结")
        print(f"{'='*80}")
        print("\n为什么需要归一化？")
        print("\n1. 拍摄高度差异导致像素尺度完全不同:")
        print("   - 10m高度: 牛在画面中更大，像素值高")
        print("   - 15m高度: 牛在画面中更小，像素值低")
        print("   - 比例差异: 1.5-2.5倍（线性特征1.5-1.6倍，面积特征2.5倍）")
        print("\n2. 绝对值无法直接比较:")
        print("   - 10m的800像素 ≠ 15m的800像素")
        print("   - 它们代表的实际物理尺寸相同")
        print("   - 直接比较会得出错误结论")
        print("\n3. 归一化的好处:")
        print("   ✓ 消除高度带来的系统性偏差")
        print("   ✓ 可以比较生长趋势（相对变化）")
        print("   ✓ 可以评估测量稳定性（变化率的规律性）")
        print("   ✓ 适用于纵向研究（同一群体随时间变化）")
        print("\n4. 推荐的归一化方法:")
        print("   方法1: 相对于首次测量的增长率 = (当前值 - 初始值) / 初始值 × 100%")
        print("   方法2: 相对于该高度均值的偏差 = (当前值 - 均值) / 均值 × 100%")
        print("   方法3: Z-score标准化 = (当前值 - 均值) / 标准差")
        print("\n5. 本研究中的应用:")
        print("   - W/L比值: 已经是归一化特征（宽度/长度），无需再归一化")
        print("   - 面积、长度、宽度: 建议使用相对增长率")
        print("   - 时序分析: 可用归一化后的数据评估生长曲线")
        
        self.analysis_results['normalization'] = normalization_results
        
    def _calculate_temporal_stats(self, df, feature_col, height_name):
        """计算特征的时序统计"""
        # 按时间点分组计算均值
        temporal_stats = df.groupby('Day_Index')[feature_col].mean().reset_index()
        temporal_stats = temporal_stats.sort_values('Day_Index')
        
        values = temporal_stats[feature_col].values
        
        # 绝对值统计
        mean_value = values.mean()
        
        # 计算总增长率（相对于第一个时间点）
        if len(values) > 0:
            total_growth = (values[-1] - values[0]) / values[0] * 100
        else:
            total_growth = 0
        
        # 时序变异系数
        temporal_cv = (values.std() / values.mean() * 100) if values.mean() > 0 else 0
        
        # 相邻点变化率
        if len(values) > 1:
            changes = np.diff(values)
            change_rates = np.abs(changes / values[:-1] * 100)
            avg_change_rate = change_rates.mean()
            change_rate_std = change_rates.std()
        else:
            avg_change_rate = 0
            change_rate_std = 0
        
        # 归一化序列（相对于第一个值）
        normalized_values = (values / values[0] * 100) if len(values) > 0 and values[0] > 0 else values
        
        return {
            'mean_value': mean_value,
            'total_growth': total_growth,
            'temporal_cv': temporal_cv,
            'avg_change_rate': avg_change_rate,
            'change_rate_std': change_rate_std,
            'values': values,
            'normalized_values': normalized_values,
            'day_indices': temporal_stats['Day_Index'].values
        }
    
    def plot_filter_comparison(self):
        """绘制过滤前后的数据对比图"""
        print("生成图表: 过滤前后数据对比...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 总体过滤效果
        ax = axes[0, 0]
        stats_10m = self.filter_stats_df[self.filter_stats_df['Height'] == '10m']
        stats_15m = self.filter_stats_df[self.filter_stats_df['Height'] == '15m']
        
        total_before_10m = stats_10m['Before'].sum()
        total_after_10m = stats_10m['After'].sum()
        total_before_15m = stats_15m['Before'].sum()
        total_after_15m = stats_15m['After'].sum()
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [total_before_10m, total_before_15m], width,
               label='Before Filter', color='#FF6B6B', alpha=0.8)
        ax.bar(x + width/2, [total_after_10m, total_after_15m], width,
               label='After Filter', color='#4ECDC4', alpha=0.8)
        
        # 添加数值标签
        for i, (before, after) in enumerate([(total_before_10m, total_after_10m),
                                              (total_before_15m, total_after_15m)]):
            retention = after / before * 100
            ax.text(i, max(before, after) + 100, f'{retention:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Instances', fontsize=11)
        ax.set_title('Total Data: Before vs After Filtering', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['10m', '15m'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. 10m各时间点对比
        ax = axes[0, 1]
        stats_10m_sorted = stats_10m.sort_values('Date')
        x_pos = np.arange(len(stats_10m_sorted))
        
        ax.bar(x_pos - width/2, stats_10m_sorted['Before'], width,
               label='Before', color='#FF6B6B', alpha=0.8)
        ax.bar(x_pos + width/2, stats_10m_sorted['After'], width,
               label='After', color='#4ECDC4', alpha=0.8)
        
        ax.set_ylabel('Number of Instances', fontsize=11)
        ax.set_title('10m Height: Time-wise Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_10m_sorted['Date'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 15m各时间点对比
        ax = axes[1, 0]
        stats_15m_sorted = stats_15m.sort_values('Date')
        x_pos = np.arange(len(stats_15m_sorted))
        
        ax.bar(x_pos - width/2, stats_15m_sorted['Before'], width,
               label='Before', color='#FF6B6B', alpha=0.8)
        ax.bar(x_pos + width/2, stats_15m_sorted['After'], width,
               label='After', color='#4ECDC4', alpha=0.8)
        
        ax.set_ylabel('Number of Instances', fontsize=11)
        ax.set_title('15m Height: Time-wise Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_15m_sorted['Date'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 保留率对比
        ax = axes[1, 1]
        
        all_dates = sorted(set(stats_10m['Date'].tolist() + stats_15m['Date'].tolist()))
        retention_10m = []
        retention_15m = []
        
        for date in all_dates:
            r10 = stats_10m[stats_10m['Date'] == date]['Retention'].values
            r15 = stats_15m[stats_15m['Date'] == date]['Retention'].values
            retention_10m.append(r10[0] if len(r10) > 0 else 0)
            retention_15m.append(r15[0] if len(r15) > 0 else 0)
        
        x_pos = np.arange(len(all_dates))
        ax.plot(x_pos, retention_10m, marker='o', linewidth=2, markersize=6,
               color='#FF6B6B', label='10m')
        ax.plot(x_pos, retention_15m, marker='s', linewidth=2, markersize=6,
               color='#4ECDC4', label='15m')
        
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Retention Rate (%)', fontsize=11)
        ax.set_title('Retention Rate by Date', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_dates, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'filter_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_path.name}")
        plt.close()
        
    def plot_pixel_distribution(self):
        """绘制像素分布对比图"""
        print("生成图表: 像素分布对比...")
        
        features = [
            ('Area_pixels', 'Area (pixels)'),
            ('MinRect_Length_pixels', 'MinRect Length (pixels)'),
            ('MinRect_Width_pixels', 'MinRect Width (pixels)'),
            ('Head_Tail_Distance_pixels', 'Head-Tail Distance (pixels)')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (feature, title) in enumerate(features):
            ax = axes[idx]
            
            data_10m = self.df_10m[feature].dropna()
            data_15m = self.df_15m[feature].dropna()
            
            # 箱线图
            bp = ax.boxplot([data_10m, data_15m], 
                            labels=['10m', '15m'],
                            patch_artist=True,
                            showfliers=False)
            
            # 设置颜色
            colors = ['#FF6B6B', '#4ECDC4']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 添加均值点
            means = [data_10m.mean(), data_15m.mean()]
            ax.plot([1, 2], means, 'D', color='red', markersize=8, label='Mean', zorder=3)
            
            # 添加统计信息
            cv_10m = data_10m.std() / data_10m.mean() * 100
            cv_15m = data_15m.std() / data_15m.mean() * 100
            
            ax.text(0.02, 0.98, 
                   f'10m: CV={cv_10m:.1f}%\n15m: CV={cv_15m:.1f}%',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
            
            ax.set_ylabel(title, fontsize=11)
            ax.set_xlabel('Height', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=9)
        
        plt.suptitle('Pixel Scale Distribution Comparison (10m vs 15m)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'pixel_distribution_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_path.name}")
        plt.close()
        
    def plot_wl_comparison(self):
        """绘制两种W/L定义的对比图"""
        print("生成图表: W/L定义对比...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. CV对比柱状图
        ax = axes[0, 0]
        wl_comp = self.analysis_results['wl_comparison']
        
        x = np.arange(2)
        width = 0.35
        
        cv_10m_orig = wl_comp[(wl_comp['Height']=='10m') & (wl_comp['WL_Type']=='Original (W/L)')]['CV'].values[0]
        cv_10m_new = wl_comp[(wl_comp['Height']=='10m') & (wl_comp['WL_Type']=='New (W/HTD)')]['CV'].values[0]
        cv_15m_orig = wl_comp[(wl_comp['Height']=='15m') & (wl_comp['WL_Type']=='Original (W/L)')]['CV'].values[0]
        cv_15m_new = wl_comp[(wl_comp['Height']=='15m') & (wl_comp['WL_Type']=='New (W/HTD)')]['CV'].values[0]
        
        ax.bar(x - width/2, [cv_10m_orig, cv_15m_orig], width, 
               label='Original (W/L)', color='#FF6B6B', alpha=0.8)
        ax.bar(x + width/2, [cv_10m_new, cv_15m_new], width, 
               label='New (W/HTD)', color='#4ECDC4', alpha=0.8)
        
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=11)
        ax.set_title('W/L Stability Comparison (CV)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['10m', '15m'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. 改进率
        ax = axes[0, 1]
        improvements = []
        for height in ['10m', '15m']:
            cv_orig = wl_comp[(wl_comp['Height']==height) & (wl_comp['WL_Type']=='Original (W/L)')]['CV'].values[0]
            cv_new = wl_comp[(wl_comp['Height']==height) & (wl_comp['WL_Type']=='New (W/HTD)')]['CV'].values[0]
            improvement = (cv_orig - cv_new) / cv_orig * 100
            improvements.append(improvement)
        
        colors_imp = ['green' if x > 0 else 'red' for x in improvements]
        ax.bar(['10m', '15m'], improvements, color=colors_imp, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Improvement (%)', fontsize=11)
        ax.set_title('CV Improvement (Original → New)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 分布对比 - 10m
        ax = axes[1, 0]
        df_10m_standing = self.df_10m  # 数据已过滤
        ax.hist([df_10m_standing['WL_Ratio'].dropna(), 
                df_10m_standing['WL_New'].dropna()],
                bins=30, label=['Original (W/L)', 'New (W/HTD)'],
                alpha=0.6, color=['#FF6B6B', '#4ECDC4'])
        ax.set_xlabel('W/L Ratio', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('10m Height - W/L Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 分布对比 - 15m
        ax = axes[1, 1]
        df_15m_standing = self.df_15m  # 数据已过滤
        ax.hist([df_15m_standing['WL_Ratio'].dropna(), 
                df_15m_standing['WL_New'].dropna()],
                bins=30, label=['Original (W/L)', 'New (W/HTD)'],
                alpha=0.6, color=['#FF6B6B', '#4ECDC4'])
        ax.set_xlabel('W/L Ratio', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('15m Height - W/L Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'wl_definition_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_path.name}")
        plt.close()
        
    def plot_temporal_stability(self):
        """绘制时间序列稳定性图 - 重点展示每个时间点的变化和规律性"""
        print("生成图表: 时间序列变化规律性...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        # 获取数据
        data_10m = self.analysis_results['temporal_stability']['10m']
        data_15m = self.analysis_results['temporal_stability']['15m']
        
        # 1. W/L时间序列对比（Original）
        ax1 = fig.add_subplot(gs[0, 0])
        days_10m = data_10m['days']
        days_15m = data_15m['days']
        
        ax1.plot(days_10m, data_10m['wl_original_means'], marker='o', linewidth=2, 
                markersize=8, color='#FF6B6B', label='10m', alpha=0.8)
        ax1.plot(days_15m, data_15m['wl_original_means'], marker='s', linewidth=2, 
                markersize=8, color='#4ECDC4', label='15m', alpha=0.8)
        
        ax1.set_xlabel('Day Index', fontsize=11)
        ax1.set_ylabel('W/L Ratio (Original)', fontsize=11)
        ax1.set_title('Time Series: Original W/L (Each Time Point)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. W/L时间序列对比（New）
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(days_10m, data_10m['wl_new_means'], marker='o', linewidth=2, 
                markersize=8, color='#FF6B6B', label='10m', alpha=0.8)
        ax2.plot(days_15m, data_15m['wl_new_means'], marker='s', linewidth=2, 
                markersize=8, color='#4ECDC4', label='15m', alpha=0.8)
        
        ax2.set_xlabel('Day Index', fontsize=11)
        ax2.set_ylabel('W/L Ratio (New)', fontsize=11)
        ax2.set_title('Time Series: New W/L (Each Time Point)', 
                     fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 相邻点变化率（Original W/L）
        ax3 = fig.add_subplot(gs[1, 0])
        days_diff_10m = days_10m[1:]  # 差分后少一个点
        days_diff_15m = days_15m[1:]
        
        ax3.plot(days_diff_10m, data_10m['wl_original_pct_change'], marker='o', 
                linewidth=2, markersize=6, color='#FF6B6B', label='10m', alpha=0.8)
        ax3.plot(days_diff_15m, data_15m['wl_original_pct_change'], marker='s', 
                linewidth=2, markersize=6, color='#4ECDC4', label='15m', alpha=0.8)
        
        # 添加平均线
        ax3.axhline(y=data_10m['stats']['original_pct_change'], color='#FF6B6B', 
                   linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.axhline(y=data_15m['stats']['original_pct_change'], color='#4ECDC4', 
                   linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax3.set_xlabel('Day Index (Between Points)', fontsize=11)
        ax3.set_ylabel('Change Rate (%)', fontsize=11)
        ax3.set_title('Adjacent Point Change Rate: Original W/L\n(Lower & More Stable = Better)', 
                     fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 添加文本说明
        ax3.text(0.02, 0.98, 
                f'10m avg: {data_10m["stats"]["original_pct_change"]:.3f}%, std: {data_10m["stats"]["original_change_std"]:.3f}%\n'
                f'15m avg: {data_15m["stats"]["original_pct_change"]:.3f}%, std: {data_15m["stats"]["original_change_std"]:.3f}%',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9)
        
        # 4. 相邻点变化率（New W/L）
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(days_diff_10m, data_10m['wl_new_pct_change'], marker='o', 
                linewidth=2, markersize=6, color='#FF6B6B', label='10m', alpha=0.8)
        ax4.plot(days_diff_15m, data_15m['wl_new_pct_change'], marker='s', 
                linewidth=2, markersize=6, color='#4ECDC4', label='15m', alpha=0.8)
        
        # 添加平均线
        ax4.axhline(y=data_10m['stats']['new_pct_change'], color='#FF6B6B', 
                   linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.axhline(y=data_15m['stats']['new_pct_change'], color='#4ECDC4', 
                   linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax4.set_xlabel('Day Index (Between Points)', fontsize=11)
        ax4.set_ylabel('Change Rate (%)', fontsize=11)
        ax4.set_title('Adjacent Point Change Rate: New W/L\n(Lower & More Stable = Better)', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 添加文本说明
        ax4.text(0.02, 0.98, 
                f'10m avg: {data_10m["stats"]["new_pct_change"]:.3f}%, std: {data_10m["stats"]["new_change_std"]:.3f}%\n'
                f'15m avg: {data_15m["stats"]["new_pct_change"]:.3f}%, std: {data_15m["stats"]["new_change_std"]:.3f}%',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9)
        
        # 5. Width变化趋势
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(days_10m, data_10m['width_means'], marker='o', linewidth=2, 
                markersize=8, color='#FF6B6B', label='10m Width', alpha=0.8)
        ax5.plot(days_15m, data_15m['width_means'], marker='s', linewidth=2, 
                markersize=8, color='#4ECDC4', label='15m Width', alpha=0.8)
        
        ax5.set_xlabel('Day Index', fontsize=11)
        ax5.set_ylabel('MinRect Width (pixels)', fontsize=11)
        ax5.set_title('Width Trend Over Time', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. Length变化趋势
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(days_10m, data_10m['length_means'], marker='o', linewidth=2, 
                markersize=8, color='#FF6B6B', label='10m Length', alpha=0.8)
        ax6.plot(days_15m, data_15m['length_means'], marker='s', linewidth=2, 
                markersize=8, color='#4ECDC4', label='15m Length', alpha=0.8)
        
        ax6.set_xlabel('Day Index', fontsize=11)
        ax6.set_ylabel('MinRect Length (pixels)', fontsize=11)
        ax6.set_title('Length Trend Over Time', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'temporal_stability_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_path.name}")
        plt.close()
        
    def plot_scale_vs_stability(self):
        """绘制尺度vs稳定性图"""
        print("生成图表: 尺度 vs 稳定性...")
        
        df_standing = self.df_combined.copy()  # 数据已过滤
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, wl_type in enumerate(['WL_Ratio', 'WL_New']):
            ax = axes[idx]
            
            for height, color, marker in [('10m', '#FF6B6B', 'o'), 
                                          ('15m', '#4ECDC4', 's')]:
                df_height = df_standing[df_standing['Height'] == height]
                
                # 按日期分组
                grouped = df_height.groupby('Date').agg({
                    'Area_pixels': 'mean',
                    wl_type: lambda x: x.std() / x.mean() * 100 if len(x) > 1 else np.nan
                }).reset_index()
                
                grouped = grouped.dropna()
                
                if len(grouped) > 0:
                    ax.scatter(grouped['Area_pixels'], grouped[wl_type],
                             alpha=0.6, s=80, color=color, marker=marker,
                             label=height, edgecolors='black', linewidth=0.5)
                    
                    # 添加趋势线
                    if len(grouped) > 3:
                        z = np.polyfit(grouped['Area_pixels'], grouped[wl_type], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(grouped['Area_pixels'].min(), 
                                            grouped['Area_pixels'].max(), 100)
                        ax.plot(x_trend, p(x_trend), '--', color=color, 
                               alpha=0.5, linewidth=2)
            
            title = 'Original W/L' if wl_type == 'WL_Ratio' else 'New W/L (W/HTD)'
            ax.set_xlabel('Mean Area (pixels)', fontsize=11)
            ax.set_ylabel('CV of W/L (%)', fontsize=11)
            ax.set_title(f'{title}: Scale vs Stability', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'scale_vs_stability.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {output_path.name}")
        plt.close()
        
    def plot_comprehensive_features_temporal(self):
        """绘制各种特征的时序变化和分布"""
        print("生成图表: 综合特征时序变化...")
        
        # 定义要分析的特征
        features = [
            ('Area_pixels', 'Area', 'pixels²'),
            ('MinRect_Length_pixels', 'Min Rect Length', 'pixels'),
            ('MinRect_Width_pixels', 'Min Rect Width', 'pixels'),
            ('Head_Tail_Distance_pixels', 'Head-Tail Distance', 'pixels'),
            ('Perimeter_pixels', 'Perimeter', 'pixels'),
            ('BBox_Width_pixels', 'BBox Width', 'pixels'),
            ('BBox_Height_pixels', 'BBox Height', 'pixels'),
        ]
        
        # 为每个特征创建单独的图表
        for feature_col, feature_name, unit in features:
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
            
            fig.suptitle(f'{feature_name} - Temporal Analysis', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            # 准备10m和15m的数据
            df_10m = self.df_10m.copy()
            df_15m = self.df_15m.copy()
            
            # 按时间点统计
            stats_10m = df_10m.groupby('Day_Index')[feature_col].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('median', 'median'),
                ('q25', lambda x: x.quantile(0.25)),
                ('q75', lambda x: x.quantile(0.75)),
                ('count', 'count')
            ]).reset_index()
            
            stats_15m = df_15m.groupby('Day_Index')[feature_col].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('median', 'median'),
                ('q25', lambda x: x.quantile(0.25)),
                ('q75', lambda x: x.quantile(0.75)),
                ('count', 'count')
            ]).reset_index()
            
            # 1. 时序均值变化（带标准差）
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.errorbar(stats_10m['Day_Index'], stats_10m['mean'], 
                        yerr=stats_10m['std'], marker='o', linewidth=2, 
                        markersize=8, color='#FF6B6B', label='10m', 
                        alpha=0.8, capsize=5, capthick=2)
            ax1.errorbar(stats_15m['Day_Index'], stats_15m['mean'], 
                        yerr=stats_15m['std'], marker='s', linewidth=2, 
                        markersize=8, color='#4ECDC4', label='15m', 
                        alpha=0.8, capsize=5, capthick=2)
            ax1.set_xlabel('Day Index', fontsize=11)
            ax1.set_ylabel(f'{feature_name} ({unit})', fontsize=11)
            ax1.set_title('Mean ± Std Over Time', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 2. 时序中位数变化（带四分位数范围）
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(stats_10m['Day_Index'], stats_10m['median'], 
                    marker='o', linewidth=2, markersize=8, 
                    color='#FF6B6B', label='10m median', alpha=0.8)
            ax2.fill_between(stats_10m['Day_Index'], stats_10m['q25'], 
                            stats_10m['q75'], alpha=0.2, color='#FF6B6B')
            ax2.plot(stats_15m['Day_Index'], stats_15m['median'], 
                    marker='s', linewidth=2, markersize=8, 
                    color='#4ECDC4', label='15m median', alpha=0.8)
            ax2.fill_between(stats_15m['Day_Index'], stats_15m['q25'], 
                            stats_15m['q75'], alpha=0.2, color='#4ECDC4')
            ax2.set_xlabel('Day Index', fontsize=11)
            ax2.set_ylabel(f'{feature_name} ({unit})', fontsize=11)
            ax2.set_title('Median with IQR Over Time', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # 3. 变异系数时序变化
            ax3 = fig.add_subplot(gs[0, 2])
            cv_10m = (stats_10m['std'] / stats_10m['mean'] * 100).values
            cv_15m = (stats_15m['std'] / stats_15m['mean'] * 100).values
            ax3.plot(stats_10m['Day_Index'], cv_10m, 
                    marker='o', linewidth=2, markersize=8, 
                    color='#FF6B6B', label='10m CV', alpha=0.8)
            ax3.plot(stats_15m['Day_Index'], cv_15m, 
                    marker='s', linewidth=2, markersize=8, 
                    color='#4ECDC4', label='15m CV', alpha=0.8)
            ax3.set_xlabel('Day Index', fontsize=11)
            ax3.set_ylabel('CV (%)', fontsize=11)
            ax3.set_title('Coefficient of Variation Over Time', 
                         fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # 4. 相邻时间点变化率
            ax4 = fig.add_subplot(gs[1, 0])
            if len(stats_10m) > 1:
                change_rate_10m = np.abs(np.diff(stats_10m['mean'].values)) / stats_10m['mean'].values[:-1] * 100
                ax4.plot(stats_10m['Day_Index'].values[1:], change_rate_10m, 
                        marker='o', linewidth=2, markersize=8, 
                        color='#FF6B6B', label='10m', alpha=0.8)
            if len(stats_15m) > 1:
                change_rate_15m = np.abs(np.diff(stats_15m['mean'].values)) / stats_15m['mean'].values[:-1] * 100
                ax4.plot(stats_15m['Day_Index'].values[1:], change_rate_15m, 
                        marker='s', linewidth=2, markersize=8, 
                        color='#4ECDC4', label='15m', alpha=0.8)
            ax4.set_xlabel('Day Index', fontsize=11)
            ax4.set_ylabel('Change Rate (%)', fontsize=11)
            ax4.set_title('Adjacent Point Change Rate', 
                         fontsize=12, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # 5. 分布对比（箱线图）
            ax5 = fig.add_subplot(gs[1, 1])
            data_to_plot = []
            labels_to_plot = []
            colors_to_plot = []
            
            # 随机采样以避免过多数据点
            sample_size = min(500, len(df_10m))
            df_10m_sample = df_10m.sample(n=sample_size, random_state=42) if len(df_10m) > sample_size else df_10m
            df_15m_sample = df_15m.sample(n=sample_size, random_state=42) if len(df_15m) > sample_size else df_15m
            
            data_to_plot.append(df_10m_sample[feature_col].values)
            labels_to_plot.append('10m')
            colors_to_plot.append('#FF6B6B')
            
            data_to_plot.append(df_15m_sample[feature_col].values)
            labels_to_plot.append('15m')
            colors_to_plot.append('#4ECDC4')
            
            bp = ax5.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                            showfliers=False, widths=0.6)
            for patch, color in zip(bp['boxes'], colors_to_plot):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax5.set_ylabel(f'{feature_name} ({unit})', fontsize=11)
            ax5.set_title('Distribution Comparison (Box Plot)', 
                         fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 6. 核密度估计分布
            ax6 = fig.add_subplot(gs[1, 2])
            from scipy import stats as scipy_stats
            
            # 10m
            if len(df_10m[feature_col].dropna()) > 10:
                density_10m = scipy_stats.gaussian_kde(df_10m[feature_col].dropna())
                x_10m = np.linspace(df_10m[feature_col].min(), 
                                   df_10m[feature_col].max(), 200)
                ax6.fill_between(x_10m, density_10m(x_10m), alpha=0.3, 
                                color='#FF6B6B', label='10m')
                ax6.plot(x_10m, density_10m(x_10m), linewidth=2, 
                        color='#FF6B6B')
            
            # 15m
            if len(df_15m[feature_col].dropna()) > 10:
                density_15m = scipy_stats.gaussian_kde(df_15m[feature_col].dropna())
                x_15m = np.linspace(df_15m[feature_col].min(), 
                                   df_15m[feature_col].max(), 200)
                ax6.fill_between(x_15m, density_15m(x_15m), alpha=0.3, 
                                color='#4ECDC4', label='15m')
                ax6.plot(x_15m, density_15m(x_15m), linewidth=2, 
                        color='#4ECDC4')
            
            ax6.set_xlabel(f'{feature_name} ({unit})', fontsize=11)
            ax6.set_ylabel('Density', fontsize=11)
            ax6.set_title('Distribution (KDE)', fontsize=12, fontweight='bold')
            ax6.legend(fontsize=10)
            ax6.grid(True, alpha=0.3)
            
            # 保存图表
            safe_name = feature_name.replace(' ', '_').replace('-', '_').lower()
            output_path = self.output_dir / f'feature_temporal_{safe_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 保存: {output_path.name}")
            plt.close()
    
    def plot_normalization_comparison(self):
        """绘制归一化前后的对比图"""
        if 'normalization' not in self.analysis_results:
            return
        
        print("生成图表: 归一化对比分析...")
        
        norm_results = self.analysis_results['normalization']
        
        # 为每个特征生成对比图
        for feature_name, data in norm_results.items():
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
            
            fig.suptitle(f'{feature_name} - Normalization Analysis', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            result_10m = data['10m']
            result_15m = data['15m']
            ratio = data['ratio']
            
            # 1. 绝对值时序对比（显示尺度差异）
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(result_10m['day_indices'], result_10m['values'], 
                    marker='o', linewidth=2, markersize=8, 
                    color='#FF6B6B', label=f'10m (mean={result_10m["mean_value"]:.0f})', 
                    alpha=0.8)
            ax1.plot(result_15m['day_indices'], result_15m['values'], 
                    marker='s', linewidth=2, markersize=8, 
                    color='#4ECDC4', label=f'15m (mean={result_15m["mean_value"]:.0f})', 
                    alpha=0.8)
            ax1.set_xlabel('Day Index', fontsize=11)
            ax1.set_ylabel(f'{feature_name} (pixels)', fontsize=11)
            ax1.set_title(f'Absolute Values (10m/15m = {ratio:.2f}x)', 
                         fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.text(0.02, 0.98, '⚠️ 无法直接比较\n尺度差异大', 
                    transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='yellow', alpha=0.3))
            
            # 2. 归一化时序对比（相对于首次测量=100）
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(result_10m['day_indices'], result_10m['normalized_values'], 
                    marker='o', linewidth=2, markersize=8, 
                    color='#FF6B6B', label=f'10m (growth={result_10m["total_growth"]:.1f}%)', 
                    alpha=0.8)
            ax2.plot(result_15m['day_indices'], result_15m['normalized_values'], 
                    marker='s', linewidth=2, markersize=8, 
                    color='#4ECDC4', label=f'15m (growth={result_15m["total_growth"]:.1f}%)', 
                    alpha=0.8)
            ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Day Index', fontsize=11)
            ax2.set_ylabel('Normalized Value (first=100)', fontsize=11)
            ax2.set_title('Normalized Growth (Relative to First Measurement)', 
                         fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.text(0.02, 0.98, '✓ 可以直接比较\n生长趋势一致', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='lightgreen', alpha=0.3))
            
            # 3. 相邻点变化率对比
            ax3 = fig.add_subplot(gs[0, 2])
            if len(result_10m['values']) > 1:
                change_rates_10m = np.abs(np.diff(result_10m['values']) / result_10m['values'][:-1] * 100)
                ax3.plot(result_10m['day_indices'][1:], change_rates_10m, 
                        marker='o', linewidth=2, markersize=8, 
                        color='#FF6B6B', label=f'10m (avg={result_10m["avg_change_rate"]:.2f}%)', 
                        alpha=0.8)
            if len(result_15m['values']) > 1:
                change_rates_15m = np.abs(np.diff(result_15m['values']) / result_15m['values'][:-1] * 100)
                ax3.plot(result_15m['day_indices'][1:], change_rates_15m, 
                        marker='s', linewidth=2, markersize=8, 
                        color='#4ECDC4', label=f'15m (avg={result_15m["avg_change_rate"]:.2f}%)', 
                        alpha=0.8)
            ax3.set_xlabel('Day Index', fontsize=11)
            ax3.set_ylabel('Change Rate (%)', fontsize=11)
            ax3.set_title('Adjacent Point Change Rate', 
                         fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # 4. 双轴对比（左轴10m，右轴15m）- 显示尺度差异
            ax4 = fig.add_subplot(gs[1, 0])
            ax4_right = ax4.twinx()
            
            line1 = ax4.plot(result_10m['day_indices'], result_10m['values'], 
                            marker='o', linewidth=2, markersize=8, 
                            color='#FF6B6B', label='10m (left axis)', alpha=0.8)
            line2 = ax4_right.plot(result_15m['day_indices'], result_15m['values'], 
                                  marker='s', linewidth=2, markersize=8, 
                                  color='#4ECDC4', label='15m (right axis)', alpha=0.8)
            
            ax4.set_xlabel('Day Index', fontsize=11)
            ax4.set_ylabel('10m Value (pixels)', fontsize=11, color='#FF6B6B')
            ax4_right.set_ylabel('15m Value (pixels)', fontsize=11, color='#4ECDC4')
            ax4.set_title('Dual-Axis Comparison (Different Scales)', 
                         fontsize=12, fontweight='bold')
            ax4.tick_params(axis='y', labelcolor='#FF6B6B')
            ax4_right.tick_params(axis='y', labelcolor='#4ECDC4')
            ax4.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left', fontsize=9)
            
            # 5. CV时序对比
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.bar([0, 1], [result_10m['temporal_cv'], result_15m['temporal_cv']], 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.7, width=0.6)
            ax5.set_xticks([0, 1])
            ax5.set_xticklabels(['10m', '15m'])
            ax5.set_ylabel('Temporal CV (%)', fontsize=11)
            ax5.set_title('Temporal Variability', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 标注数值
            for i, (val, name) in enumerate([(result_10m['temporal_cv'], '10m'), 
                                             (result_15m['temporal_cv'], '15m')]):
                ax5.text(i, val + 0.2, f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
            
            # 6. 总增长率对比
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.bar([0, 1], [result_10m['total_growth'], result_15m['total_growth']], 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.7, width=0.6)
            ax6.set_xticks([0, 1])
            ax6.set_xticklabels(['10m', '15m'])
            ax6.set_ylabel('Total Growth (%)', fontsize=11)
            ax6.set_title('Total Growth (First to Last)', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # 标注数值
            for i, (val, name) in enumerate([(result_10m['total_growth'], '10m'), 
                                             (result_15m['total_growth'], '15m')]):
                y_pos = val + (1 if val > 0 else -1)
                ax6.text(i, y_pos, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
            
            # 保存图表
            safe_name = feature_name.replace(' ', '_').replace('-', '_').lower()
            output_path = self.output_dir / f'normalization_{safe_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 保存: {output_path.name}")
            plt.close()
    
    def save_summary_report(self):
        """保存总结报告"""
        print("\n保存分析报告...")
        
        report_path = self.output_dir / 'pixel_stability_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("牛实例像素尺度与W/L稳定性分析报告（严格筛选版）\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"10m数据: {self.csv_10m}\n")
            f.write(f"15m数据: {self.csv_15m}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            # 筛选条件
            f.write("=" * 80 + "\n")
            f.write("数据筛选条件\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. Class == 'standing'\n")
            f.write("2. ROI_Region == 'center_70'\n")
            f.write("3. 15m高度: 490 < MinRect_Length_pixels < 600\n")
            f.write("4. 10m高度: 790 < MinRect_Length_pixels < 950\n\n")
            
            # 过滤统计
            f.write("=" * 80 + "\n")
            f.write("数据过滤统计\n")
            f.write("=" * 80 + "\n\n")
            
            stats_10m = self.filter_stats_df[self.filter_stats_df['Height'] == '10m']
            stats_15m = self.filter_stats_df[self.filter_stats_df['Height'] == '15m']
            
            f.write("10m高度:\n")
            f.write(f"  原始数据: {stats_10m['Before'].sum()} 条\n")
            f.write(f"  过滤后: {stats_10m['After'].sum()} 条\n")
            f.write(f"  过滤掉: {stats_10m['Filtered'].sum()} 条\n")
            f.write(f"  保留率: {stats_10m['After'].sum()/stats_10m['Before'].sum()*100:.1f}%\n\n")
            
            f.write("15m高度:\n")
            f.write(f"  原始数据: {stats_15m['Before'].sum()} 条\n")
            f.write(f"  过滤后: {stats_15m['After'].sum()} 条\n")
            f.write(f"  过滤掉: {stats_15m['Filtered'].sum()} 条\n")
            f.write(f"  保留率: {stats_15m['After'].sum()/stats_15m['Before'].sum()*100:.1f}%\n\n")
            
            # 各时间点详细统计
            f.write("各时间点过滤统计:\n\n")
            f.write("10m高度:\n")
            f.write("  日期          过滤前  过滤后  过滤掉  保留率\n")
            f.write("  " + "-" * 50 + "\n")
            for _, row in stats_10m.iterrows():
                f.write(f"  {row['Date']}  {row['Before']:6d}  {row['After']:6d}  "
                       f"{row['Filtered']:6d}  {row['Retention']:6.1f}%\n")
            
            f.write("\n15m高度:\n")
            f.write("  日期          过滤前  过滤后  过滤掉  保留率\n")
            f.write("  " + "-" * 50 + "\n")
            for _, row in stats_15m.iterrows():
                f.write(f"  {row['Date']}  {row['Before']:6d}  {row['After']:6d}  "
                       f"{row['Filtered']:6d}  {row['Retention']:6.1f}%\n")
            
            f.write("\n")
            
            # 像素尺度统计
            f.write("=" * 80 + "\n")
            f.write("1. 像素尺度统计对比\n")
            f.write("=" * 80 + "\n\n")
            
            if 'pixel_scale' in self.analysis_results:
                for feature, stats in self.analysis_results['pixel_scale'].items():
                    f.write(f"{feature}:\n")
                    f.write(f"  10m: 均值={stats['10m']['mean']:.1f}, "
                           f"标准差={stats['10m']['std']:.1f}, CV={stats['10m']['cv']:.2f}%\n")
                    f.write(f"  15m: 均值={stats['15m']['mean']:.1f}, "
                           f"标准差={stats['15m']['std']:.1f}, CV={stats['15m']['cv']:.2f}%\n")
                    f.write(f"  均值比值(10m/15m): {stats['10m']['mean']/stats['15m']['mean']:.3f}\n")
                    f.write(f"  CV比值(10m/15m): {stats['10m']['cv']/stats['15m']['cv']:.3f}\n\n")
            
            # W/L对比
            f.write("=" * 80 + "\n")
            f.write("2. W/L定义对比\n")
            f.write("=" * 80 + "\n\n")
            
            if 'wl_comparison' in self.analysis_results:
                wl_comp = self.analysis_results['wl_comparison']
                f.write(wl_comp.to_string(index=False))
                f.write("\n\n")
            
            # 验证结论
            f.write("=" * 80 + "\n")
            f.write("3. 验证结论\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("猜想1: 10m高度的牛像素更大但更不稳定\n")
            if 'pixel_scale' in self.analysis_results:
                area_stats = self.analysis_results['pixel_scale']['Area_pixels']
                area_ratio = area_stats['10m']['mean'] / area_stats['15m']['mean']
                cv_ratio = area_stats['10m']['cv'] / area_stats['15m']['cv']
                
                f.write(f"  结论: {'✓ 支持' if area_ratio > 1 and cv_ratio > 1 else '✗ 不支持'}\n")
                f.write(f"  10m面积是15m的 {area_ratio:.2f} 倍\n")
                f.write(f"  10m变异系数是15m的 {cv_ratio:.2f} 倍\n\n")
            
            f.write("猜想2: 新W/L定义(W/HTD)比原始定义(W/L)更稳定\n")
            if 'wl_comparison' in self.analysis_results:
                wl_comp = self.analysis_results['wl_comparison']
                
                for height in ['10m', '15m']:
                    cv_orig = wl_comp[(wl_comp['Height']==height) & 
                                     (wl_comp['WL_Type']=='Original (W/L)')]['CV'].values[0]
                    cv_new = wl_comp[(wl_comp['Height']==height) & 
                                    (wl_comp['WL_Type']=='New (W/HTD)')]['CV'].values[0]
                    improvement = (cv_orig - cv_new) / cv_orig * 100
                    
                    f.write(f"  {height}: CV改进 {improvement:+.1f}% "
                           f"{'✓' if improvement > 0 else '✗'}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"✓ 保存报告: {report_path}")
        
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n" + "=" * 80)
        print("开始像素尺度与W/L稳定性分析（严格筛选版）")
        print("=" * 80 + "\n")
        
        # 1. 加载数据（包含过滤）
        self.load_data()
        
        # 2. 运行分析
        self.analyze_pixel_scale()
        self.analyze_wl_comparison()
        self.analyze_wl_temporal_stability()
        self.analyze_scale_vs_stability()
        self.analyze_correlation()
        self.analyze_normalization_necessity()  # 新增：归一化必要性分析
        
        # 3. 生成可视化
        print("\n" + "=" * 80)
        print("生成可视化图表...")
        print("=" * 80)
        self.plot_filter_comparison()  # 首先显示过滤效果
        self.plot_pixel_distribution()
        self.plot_wl_comparison()
        self.plot_temporal_stability()
        self.plot_scale_vs_stability()
        self.plot_comprehensive_features_temporal()  # 综合特征时序分析
        self.plot_normalization_comparison()  # 新增：归一化对比分析
        
        # 4. 保存报告
        self.save_summary_report()
        
        print("\n" + "=" * 80)
        print("✅ 分析完成!")
        print("=" * 80)
        print(f"📁 所有结果已保存到: {self.output_dir}")
        print()


def main():
    """主函数"""
    # 设置数据路径
    base_dir = Path(__file__).parent.parent.parent  # 回到BeefCattleDetection目录
    csv_10m = base_dir / 'data' / 'yolo_sam_w_10m' / 'instances_results_20260112_105253' / 'instances_metadata_all.csv'
    csv_15m = base_dir / 'data' / 'yolo_sam_w_15m' / 'instances_results_20260110_172441' / 'instances_metadata_all.csv'
    
    # 检查文件是否存在
    if not csv_10m.exists():
        print(f"❌ 错误: 找不到10m数据文件: {csv_10m}")
        return
    
    if not csv_15m.exists():
        print(f"❌ 错误: 找不到15m数据文件: {csv_15m}")
        return
    
    # 创建分析器
    analyzer = PixelStabilityAnalyzer(
        csv_10m=str(csv_10m),
        csv_15m=str(csv_15m)
    )
    
    # 运行完整分析
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
