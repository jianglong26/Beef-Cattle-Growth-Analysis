
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

class WeightDataVisualizer:
    def __init__(self, data_dir):
        """
        初始化可视化器
        
        Args:
            data_dir: 数据文件所在目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载数据
        self.timeline_df = self._load_timeline_data()
        self.group_stats_df = self._load_group_statistics()
        self.overall_stats_df = self._load_overall_statistics()
        
        # 定义时间点和对应日期
        self.time_points = ['0710', '0720', '0731', '0811', '0821', '0831', '0921', '0928', '1005', '1012', '1019', '1027']
        self.time_labels = ['Jul 10', 'Jul 20', 'Jul 31', 'Aug 11', 'Aug 21', 'Aug 31', 'Sep 21', 'Sep 28', 'Oct 5', 'Oct 12', 'Oct 19', 'Oct 27']
        
        # 定义颜色方案
        self.colors = {
            'Bottom_30_Percent': '#FF6B6B',  # 红色 - 最轻30%
            'Middle_40_Percent': '#4ECDC4',  # 青色 - 中间40%
            'Top_30_Percent': '#45B7D1',    # 蓝色 - 最重30%
            'overall': '#2E8B57'             # 深绿色 - 总体
        }
    
    def _load_timeline_data(self):
        """加载时间线数据"""
        try:
            df = pd.read_csv(self.data_dir / "Cattle_Weight_Timeline.csv")
            print(f"成功加载时间线数据：{len(df)} 头牛")
            return df
        except Exception as e:
            print(f"加载时间线数据失败：{e}")
            return None
    
    def _load_group_statistics(self):
        """加载分组统计数据"""
        try:
            df = pd.read_csv(self.data_dir / "Weight_Group_Statistics_By_TimePoint.csv")
            print(f"成功加载分组统计数据：{len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载分组统计数据失败：{e}")
            return None
    
    def _load_overall_statistics(self):
        """加载总体统计数据"""
        try:
            df = pd.read_csv(self.data_dir / "Overall_Group_Statistics.csv")
            print(f"成功加载总体统计数据：{len(df)} 个分组")
            return df
        except Exception as e:
            print(f"加载总体统计数据失败：{e}")
            return None
    
    def plot_all_cattle_weight_timeline(self):
        """绘制所有牛的体重时间线图"""
        if self.timeline_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 获取权重列
        weight_cols = [f'Weight_{tp}' for tp in self.time_points]
        
        # 为每头牛绘制折线
        for idx, row in self.timeline_df.iterrows():
            weights = [row[col] for col in weight_cols]
            ax.plot(range(len(self.time_points)), weights, 
                   alpha=0.3, linewidth=0.8, color='gray')
        
        # 计算和绘制平均线
        mean_weights = [self.timeline_df[col].mean() for col in weight_cols]
        ax.plot(range(len(self.time_points)), mean_weights, 
               color='red', linewidth=3, marker='o', markersize=8, 
               label=f'Overall Average (n={len(self.timeline_df)})')
        
        # 设置图表属性
        ax.set_xlabel('Time Points', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
        ax.set_title('Individual Cattle Weight Timeline\nAll Cattle Growth Trajectories', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(self.time_points)))
        ax.set_xticklabels(self.time_labels, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # 添加统计信息
        initial_mean = mean_weights[0]
        final_mean = mean_weights[-1]
        total_gain = final_mean - initial_mean
        ax.text(0.02, 0.98, f'Average Initial Weight: {initial_mean:.1f} kg\n'
                           f'Average Final Weight: {final_mean:.1f} kg\n'
                           f'Average Total Gain: {total_gain:.1f} kg',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "all_cattle_weight_timeline.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"保存图表：all_cattle_weight_timeline.png")
    
    def plot_weight_groups_timeline(self):
        """绘制分组体重时间线图（平均值±标准差）"""
        if self.group_stats_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 为每个分组绘制折线
        for group in ['Bottom_30_Percent', 'Middle_40_Percent', 'Top_30_Percent']:
            group_data = self.group_stats_df[self.group_stats_df['Weight_Group'] == group]
            
            if len(group_data) == 0:
                continue
            
            # 按时间点排序
            group_data = group_data.sort_values('Time_Point')
            
            means = group_data['Mean_Weight_kg'].values
            stds = group_data['Std_Dev_kg'].values
            
            # 获取组名和颜色
            group_label = group.replace('_', ' ').replace('Percent', '%')
            color = self.colors[group]
            
            # 绘制平均线
            x_pos = range(len(means))
            line = ax.plot(x_pos, means, color=color, linewidth=3, 
                          marker='o', markersize=8, label=group_label)
            
            # 添加标准差区域
            ax.fill_between(x_pos, means - stds, means + stds, 
                           color=color, alpha=0.2)
            
            # 在每个点添加数值标签
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.annotate(f'{mean:.0f}±{std:.0f}', 
                           (i, mean), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
        
        # 设置图表属性
        ax.set_xlabel('Time Points', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
        ax.set_title('Weight Groups Timeline (Mean ± Standard Deviation)\nCattle Weight Development by Initial Weight Groups', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(self.time_points)))
        ax.set_xticklabels(self.time_labels, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "weight_groups_timeline.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"保存图表：weight_groups_timeline.png")
    
    def plot_smooth_average_timeline(self):
        """绘制平滑的平均体重时间线"""
        if self.timeline_df is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 获取权重列
        weight_cols = [f'Weight_{tp}' for tp in self.time_points]
        
        # 计算统计数据
        means = [self.timeline_df[col].mean() for col in weight_cols]
        stds = [self.timeline_df[col].std() for col in weight_cols]
        medians = [self.timeline_df[col].median() for col in weight_cols]
        
        x_pos = np.arange(len(self.time_points))
        
        # 第一个子图：平均体重±标准差
        ax1.plot(x_pos, means, 'o-', color='blue', linewidth=3, markersize=8, label='Mean Weight')
        ax1.fill_between(x_pos, np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds), 
                        alpha=0.3, color='blue', label='±1 Standard Deviation')
        
        # 添加平滑曲线（三次样条插值）
        from scipy.interpolate import interp1d
        x_smooth = np.linspace(0, len(self.time_points)-1, 100)
        f_smooth = interp1d(x_pos, means, kind='cubic')
        y_smooth = f_smooth(x_smooth)
        ax1.plot(x_smooth, y_smooth, '--', color='red', linewidth=2, label='Smooth Trend')
        
        ax1.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Weight Timeline with Smooth Trend', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.time_labels, rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 第二个子图：平均值vs中位数比较
        ax2.plot(x_pos, means, 'o-', color='blue', linewidth=2, markersize=6, label='Mean')
        ax2.plot(x_pos, medians, 's-', color='orange', linewidth=2, markersize=6, label='Median')
        
        # 计算日增重
        daily_gains = []
        for i in range(1, len(means)):
            days_diff = (i * 10)  # 近似天数差
            if days_diff > 0:
                gain = (means[i] - means[i-1]) / days_diff * 10  # 10天增重
                daily_gains.append(gain)
        
        ax2_twin = ax2.twinx()
        if daily_gains:
            ax2_twin.bar(x_pos[1:], daily_gains, alpha=0.3, color='green', 
                        label='10-day Weight Gain', width=0.6)
            ax2_twin.set_ylabel('Weight Gain per 10 days (kg)', fontsize=10, color='green')
        
        ax2.set_xlabel('Time Points', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
        ax2.set_title('Mean vs Median Weight Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.time_labels, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        if daily_gains:
            ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "smooth_average_timeline.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"保存图表：smooth_average_timeline.png")
    
    def plot_weight_distribution_evolution(self):
        """绘制体重分布演变图"""
        if self.timeline_df is None:
            return
        
        # 选择几个关键时间点
        key_timepoints = ['0710', '0821', '0928', '1027']
        key_labels = ['Jul 10 (Initial)', 'Aug 21', 'Sep 28', 'Oct 27 (Final)']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (tp, label) in enumerate(zip(key_timepoints, key_labels)):
            weights = self.timeline_df[f'Weight_{tp}']
            
            # 直方图
            axes[i].hist(weights, bins=20, alpha=0.7, color=plt.cm.viridis(i/3), 
                        edgecolor='black', linewidth=0.5)
            
            # 添加统计线
            mean_weight = weights.mean()
            median_weight = weights.median()
            axes[i].axvline(mean_weight, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_weight:.1f}kg')
            axes[i].axvline(median_weight, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_weight:.1f}kg')
            
            axes[i].set_title(f'{label}\nWeight Distribution', fontweight='bold')
            axes[i].set_xlabel('Weight (kg)')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Weight Distribution Evolution Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "weight_distribution_evolution.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"保存图表：weight_distribution_evolution.png")
    
    def plot_growth_performance_heatmap(self):
        """绘制生长性能热力图"""
        if self.timeline_df is None:
            return
        
        # 计算每个时间段的增重
        growth_data = []
        periods = []
        
        weight_cols = [f'Weight_{tp}' for tp in self.time_points]
        
        for i in range(len(weight_cols)-1):
            period_name = f"{self.time_labels[i]} to {self.time_labels[i+1]}"
            periods.append(period_name)
            
            period_gains = []
            for idx, row in self.timeline_df.iterrows():
                gain = row[weight_cols[i+1]] - row[weight_cols[i]]
                period_gains.append(gain)
            
            growth_data.append(period_gains)
        
        # 转换为numpy数组
        growth_matrix = np.array(growth_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 热力图1：原始数据
        im1 = ax1.imshow(growth_matrix, aspect='auto', cmap='RdYlGn')
        ax1.set_title('Individual Cattle Growth Performance Heatmap\n(Weight Gain per Period)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cattle ID (sorted by initial weight)')
        ax1.set_ylabel('Time Periods')
        ax1.set_yticks(range(len(periods)))
        ax1.set_yticklabels([p.replace(' to ', '\nto\n') for p in periods], fontsize=8)
        
        # 添加颜色条
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Weight Gain (kg)', fontsize=10)
        
        # 热力图2：分组平均
        # 按初始体重分组
        total_cattle = len(self.timeline_df)
        group_size = total_cattle // 10  # 分成10组
        
        group_averages = []
        for i in range(len(periods)):
            group_avg_row = []
            for g in range(10):
                start_idx = g * group_size
                end_idx = min((g + 1) * group_size, total_cattle)
                if start_idx < total_cattle:
                    group_avg = np.mean(growth_matrix[i, start_idx:end_idx])
                    group_avg_row.append(group_avg)
            group_averages.append(group_avg_row)
        
        group_matrix = np.array(group_averages)
        im2 = ax2.imshow(group_matrix, aspect='auto', cmap='RdYlGn')
        ax2.set_title('Average Growth Performance by Weight Groups\n(10 Groups by Initial Weight)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Weight Groups (1=Lightest, 10=Heaviest)')
        ax2.set_ylabel('Time Periods')
        ax2.set_yticks(range(len(periods)))
        ax2.set_yticklabels([p.replace(' to ', '\nto\n') for p in periods], fontsize=8)
        ax2.set_xticks(range(10))
        ax2.set_xticklabels([f'G{i+1}' for i in range(10)])
        
        # 添加数值标签
        for i in range(len(periods)):
            for j in range(min(10, len(group_averages[i]))):
                text = ax2.text(j, i, f'{group_matrix[i, j]:.1f}', 
                               ha="center", va="center", color="black", fontsize=8)
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Average Weight Gain (kg)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "growth_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"保存图表：growth_performance_heatmap.png")
    
    def plot_comprehensive_dashboard(self):
        """绘制综合仪表板"""
        if self.timeline_df is None or self.overall_stats_df is None:
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], width_ratios=[1, 1, 1])
        
        # 1. 体重时间线（左上）
        ax1 = fig.add_subplot(gs[0, :2])
        weight_cols = [f'Weight_{tp}' for tp in self.time_points]
        means = [self.timeline_df[col].mean() for col in weight_cols]
        stds = [self.timeline_df[col].std() for col in weight_cols]
        
        x_pos = np.arange(len(self.time_points))
        ax1.plot(x_pos, means, 'o-', color='blue', linewidth=3, markersize=8)
        ax1.fill_between(x_pos, np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds), alpha=0.3, color='blue')
        ax1.set_title('Average Weight Timeline', fontweight='bold', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.time_labels, rotation=45, fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 分组对比（右上）
        ax2 = fig.add_subplot(gs[0, 2])
        if self.overall_stats_df is not None:
            groups = self.overall_stats_df['Weight_Group'].str.replace('_', ' ')
            initial_means = self.overall_stats_df['Initial_Weight_Mean_kg']
            final_means = self.overall_stats_df['Final_Weight_Mean_kg']
            
            x = np.arange(len(groups))
            width = 0.35
            ax2.bar(x - width/2, initial_means, width, label='Initial', alpha=0.8)
            ax2.bar(x + width/2, final_means, width, label='Final', alpha=0.8)
            ax2.set_title('Initial vs Final Weight by Groups', fontweight='bold', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(['Bottom 30%', 'Middle 40%', 'Top 30%'], fontsize=10)
            ax2.legend()
        
        # 3. 体重分布（左中）
        ax3 = fig.add_subplot(gs[1, 0])
        initial_weights = self.timeline_df['Weight_0710']
        ax3.hist(initial_weights, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(initial_weights.mean(), color='red', linestyle='--', linewidth=2)
        ax3.set_title('Initial Weight Distribution', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Weight (kg)')
        
        # 4. 最终体重分布（中中）
        ax4 = fig.add_subplot(gs[1, 1])
        final_weights = self.timeline_df['Weight_1027']
        ax4.hist(final_weights, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(final_weights.mean(), color='red', linestyle='--', linewidth=2)
        ax4.set_title('Final Weight Distribution', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Weight (kg)')
        
        # 5. 日增重分布（右中）
        ax5 = fig.add_subplot(gs[1, 2])
        daily_gains = self.timeline_df['Average_Daily_Gain']
        ax5.hist(daily_gains, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(daily_gains.mean(), color='red', linestyle='--', linewidth=2)
        ax5.set_title('Daily Gain Distribution', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Daily Gain (kg/day)')
        
        # 6. 相关性散点图（左下）
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.scatter(self.timeline_df['Weight_0710'], self.timeline_df['Average_Daily_Gain'], 
                   alpha=0.6, s=50)
        ax6.set_xlabel('Initial Weight (kg)')
        ax6.set_ylabel('Daily Gain (kg/day)')
        ax6.set_title('Initial Weight vs Daily Gain', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # 7. 总增重散点图（中下）
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.scatter(self.timeline_df['Weight_0710'], self.timeline_df['Total_Weight_Gain'], 
                   alpha=0.6, s=50, color='green')
        ax7.set_xlabel('Initial Weight (kg)')
        ax7.set_ylabel('Total Weight Gain (kg)')
        ax7.set_title('Initial Weight vs Total Gain', fontweight='bold', fontsize=12)
        ax7.grid(True, alpha=0.3)
        
        # 8. 箱线图（右下）
        ax8 = fig.add_subplot(gs[2, 2])
        # 按初始体重分组制作箱线图
        sorted_df = self.timeline_df.sort_values('Weight_0710')
        total_cattle = len(sorted_df)
        
        bottom_30 = sorted_df.iloc[:int(total_cattle*0.3)]['Average_Daily_Gain']
        middle_40 = sorted_df.iloc[int(total_cattle*0.3):int(total_cattle*0.7)]['Average_Daily_Gain']
        top_30 = sorted_df.iloc[int(total_cattle*0.7):]['Average_Daily_Gain']
        
        ax8.boxplot([bottom_30, middle_40, top_30], labels=['Bottom 30%', 'Middle 40%', 'Top 30%'])
        ax8.set_title('Daily Gain by Weight Groups', fontweight='bold', fontsize=12)
        ax8.set_ylabel('Daily Gain (kg/day)')
        
        # 9. 统计摘要表（底部）
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')
        
        # 创建统计摘要
        summary_data = [
            ['Total Cattle', f"{len(self.timeline_df)}"],
            ['Avg Initial Weight', f"{self.timeline_df['Weight_0710'].mean():.1f} kg"],
            ['Avg Final Weight', f"{self.timeline_df['Weight_1027'].mean():.1f} kg"],
            ['Avg Total Gain', f"{self.timeline_df['Total_Weight_Gain'].mean():.1f} kg"],
            ['Avg Daily Gain', f"{self.timeline_df['Average_Daily_Gain'].mean():.3f} kg/day"],
            ['Growth Period', f"{self.timeline_df['Total_Days'].iloc[0]:.0f} days"]
        ]
        
        table = ax9.table(cellText=summary_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        ax9.set_title('Summary Statistics', fontweight='bold', fontsize=14, pad=20)
        
        plt.suptitle('Cattle Weight Growth Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"保存图表：comprehensive_dashboard.png")
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("开始生成可视化图表...")
        print(f"输出目录：{self.output_dir}")
        
        print("\n1. 生成所有牛体重时间线图...")
        self.plot_all_cattle_weight_timeline()
        
        print("\n2. 生成分组体重时间线图...")
        self.plot_weight_groups_timeline()
        
        print("\n3. 生成平滑平均体重时间线...")
        self.plot_smooth_average_timeline()
        
        print("\n4. 生成体重分布演变图...")
        self.plot_weight_distribution_evolution()
        
        print("\n5. 生成生长性能热力图...")
        self.plot_growth_performance_heatmap()
        
        print("\n6. 生成综合仪表板...")
        self.plot_comprehensive_dashboard()
        
        print(f"\n所有可视化图表已生成完成！")
        print(f"图表保存位置：{self.output_dir}")

def main():
    """主函数"""
    # 设置数据目录
    data_dir = r"D:\Datasets\CowDetection\weight_analysis_output"
    
    # 创建可视化器
    visualizer = WeightDataVisualizer(data_dir)
    
    # 生成所有可视化图表
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()