"""
训练结果可视化脚本
用于生成适合论文展示的YOLOV11训练结果图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class TrainingResultsVisualizer:
    """训练结果可视化类"""
    
    def __init__(self, csv_path, output_dir=None):
        """
        初始化
        Args:
            csv_path: CSV结果文件路径
            output_dir: 输出目录，默认为CSV文件所在目录
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 读取数据
        self.df = pd.read_csv(csv_path)
        print(f"加载数据完成: {len(self.df)} epochs")
        
    def save_figure(self, fig, name):
        """保存图表为PNG和PDF格式"""
        png_path = self.output_dir / f"{name}.png"
        pdf_path = self.output_dir / f"{name}.pdf"
        
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"保存图表: {png_path.name} 和 {pdf_path.name}")
        
    def plot_loss_curves(self):
        """
        图1: 训练和验证损失曲线
        展示box_loss, cls_loss, dfl_loss的训练和验证曲线
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        loss_types = [
            ('box_loss', 'Box Loss'),
            ('cls_loss', 'Classification Loss'),
            ('dfl_loss', 'DFL Loss')
        ]
        
        for idx, (loss_key, loss_name) in enumerate(loss_types):
            ax = axes[idx]
            
            # 绘制训练损失，线条变细
            ax.plot(self.df['epoch'], self.df[f'train/{loss_key}'], 
                   label='Training', linewidth=1.5, color='#2E86AB', alpha=0.9)
            
            # 绘制验证损失，线条变细
            ax.plot(self.df['epoch'], self.df[f'val/{loss_key}'], 
                   label='Validation', linewidth=1.5, color='#A23B72', alpha=0.9)
            
            # 增大字体
            ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=16, fontweight='bold')
            ax.set_title(loss_name, fontsize=18, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        self.save_figure(fig, '01_loss_curves')
        plt.close()
        
    def plot_metrics_curves(self):
        """
        图2: 性能指标曲线
        展示Precision, Recall, mAP50, mAP50-95
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics = [
            ('metrics/precision(B)', 'Precision', '#06D6A0'),
            ('metrics/recall(B)', 'Recall', '#118AB2'),
            ('metrics/mAP50(B)', 'mAP@0.5', '#EF476F'),
            ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', '#FFD166')
        ]
        
        # 线条更细，linewidth从2.5改为1.5
        for metric_key, metric_name, color in metrics:
            ax.plot(self.df['epoch'], self.df[metric_key], 
                   label=metric_name, linewidth=1.5, color=color, alpha=0.9)
        
        # 增大坐标轴标签字体
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=16, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=18, fontweight='bold')
        
        # 增大图例字体
        ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=14)
        
        # 增大刻度字体
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        self.save_figure(fig, '02_metrics_curves')
        plt.close()
        
    def plot_learning_rate(self):
        """
        图3: 学习率变化曲线
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.df['epoch'], self.df['lr/pg0'], 
               linewidth=1.5, color='#F77F00', alpha=0.9)
        
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=16, fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        self.save_figure(fig, '03_learning_rate')
        plt.close()
        
    def plot_comprehensive_comparison(self):
        """
        图4: 综合损失对比（训练vs验证）
        展示总体损失趋势
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：训练损失综合
        train_losses = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
        colors_train = ['#2E86AB', '#A23B72', '#F18F01']
        
        for loss, color in zip(train_losses, colors_train):
            loss_name = loss.split('/')[-1].replace('_', ' ').title()
            ax1.plot(self.df['epoch'], self.df[loss], 
                    label=loss_name, linewidth=1.5, color=color, alpha=0.9)
        
        ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=16, fontweight='bold')
        ax1.set_title('Training Losses', fontsize=18, fontweight='bold')
        ax1.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.grid(True, alpha=0.3)
        
        # 右图：验证损失综合
        val_losses = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
        colors_val = ['#06D6A0', '#EF476F', '#FFD166']
        
        for loss, color in zip(val_losses, colors_val):
            loss_name = loss.split('/')[-1].replace('_', ' ').title()
            ax2.plot(self.df['epoch'], self.df[loss], 
                    label=loss_name, linewidth=1.5, color=color, alpha=0.9)
        
        ax2.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=16, fontweight='bold')
        ax2.set_title('Validation Losses', fontsize=18, fontweight='bold')
        ax2.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, '04_loss_comparison')
        plt.close()
        
    def plot_final_metrics_bar(self):
        """
        图5: 最终性能指标柱状图
        展示最后epoch的各项指标
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        final_epoch = self.df.iloc[-1]
        
        metrics = {
            'Precision': final_epoch['metrics/precision(B)'],
            'Recall': final_epoch['metrics/recall(B)'],
            'mAP@0.5': final_epoch['metrics/mAP50(B)'],
            'mAP@0.5:0.95': final_epoch['metrics/mAP50-95(B)']
        }
        
        colors = ['#06D6A0', '#118AB2', '#EF476F', '#FFD166']
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=16, fontweight='bold')
        ax.set_title(f'Final Performance Metrics (Epoch {int(final_epoch["epoch"])})', 
                    fontsize=18, fontweight='bold')
        ax.set_ylim([0.9, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.save_figure(fig, '05_final_metrics')
        plt.close()
        
    def plot_training_time(self):
        """
        图6: 训练时间分析
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算每个epoch的时间
        time_per_epoch = self.df['time'].diff()
        time_per_epoch.iloc[0] = self.df['time'].iloc[0]
        
        ax.plot(self.df['epoch'], time_per_epoch, 
               linewidth=1.5, color='#8338EC', alpha=0.9, marker='o', 
               markersize=2, markevery=10)
        
        # 添加平均线
        mean_time = time_per_epoch.mean()
        ax.axhline(y=mean_time, color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Average: {mean_time:.2f}s')
        
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel('Time per Epoch (seconds)', fontsize=16, fontweight='bold')
        ax.set_title('Training Time per Epoch', fontsize=18, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, '06_training_time')
        plt.close()
        
    def plot_convergence_analysis(self):
        """
        图7: 收敛性分析
        展示mAP50-95的提升趋势和收敛情况
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：mAP50-95完整曲线
        ax1.plot(self.df['epoch'], self.df['metrics/mAP50-95(B)'], 
                linewidth=1.8, color='#EF476F', alpha=0.9)
        
        # 标记最佳点
        best_idx = self.df['metrics/mAP50-95(B)'].idxmax()
        best_epoch = self.df.loc[best_idx, 'epoch']
        best_map = self.df.loc[best_idx, 'metrics/mAP50-95(B)']
        
        ax1.scatter([best_epoch], [best_map], color='red', s=250, 
                   zorder=5, marker='*', edgecolors='black', linewidth=2)
        ax1.annotate(f'Best: {best_map:.4f}\nEpoch: {int(best_epoch)}',
                    xy=(best_epoch, best_map), 
                    xytext=(best_epoch + 20, best_map - 0.03),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                   color='red', lw=2))
        
        ax1.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax1.set_ylabel('mAP@0.5:0.95', fontsize=16, fontweight='bold')
        ax1.set_title('Model Convergence', fontsize=18, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.grid(True, alpha=0.3)
        
        # 右图：性能提升百分比
        initial_map = self.df['metrics/mAP50-95(B)'].iloc[0]
        improvement = (self.df['metrics/mAP50-95(B)'] - initial_map) / initial_map * 100
        
        ax2.fill_between(self.df['epoch'], 0, improvement, 
                        color='#06D6A0', alpha=0.6)
        ax2.plot(self.df['epoch'], improvement, 
                linewidth=1.8, color='#118AB2', alpha=0.9)
        
        ax2.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Improvement (%)', fontsize=16, fontweight='bold')
        ax2.set_title('Performance Improvement', fontsize=18, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        self.save_figure(fig, '07_convergence_analysis')
        plt.close()
        
    def generate_statistics_summary(self):
        """
        生成训练统计摘要文本文件
        """
        output_file = self.output_dir / 'training_statistics.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLOV11 Training Statistics Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            f.write(f"Total Epochs: {len(self.df)}\n")
            f.write(f"Total Training Time: {self.df['time'].iloc[-1]:.2f} seconds ({self.df['time'].iloc[-1]/3600:.2f} hours)\n")
            f.write(f"Average Time per Epoch: {self.df['time'].diff().mean():.2f} seconds\n\n")
            
            # 最终性能
            final = self.df.iloc[-1]
            f.write("-" * 60 + "\n")
            f.write("Final Performance Metrics (Epoch 200):\n")
            f.write("-" * 60 + "\n")
            f.write(f"Precision:       {final['metrics/precision(B)']:.6f}\n")
            f.write(f"Recall:          {final['metrics/recall(B)']:.6f}\n")
            f.write(f"mAP@0.5:         {final['metrics/mAP50(B)']:.6f}\n")
            f.write(f"mAP@0.5:0.95:    {final['metrics/mAP50-95(B)']:.6f}\n\n")
            
            # 最佳性能
            best_idx = self.df['metrics/mAP50-95(B)'].idxmax()
            best = self.df.iloc[best_idx]
            f.write("-" * 60 + "\n")
            f.write(f"Best Performance (Epoch {int(best['epoch'])}):\n")
            f.write("-" * 60 + "\n")
            f.write(f"Precision:       {best['metrics/precision(B)']:.6f}\n")
            f.write(f"Recall:          {best['metrics/recall(B)']:.6f}\n")
            f.write(f"mAP@0.5:         {best['metrics/mAP50(B)']:.6f}\n")
            f.write(f"mAP@0.5:0.95:    {best['metrics/mAP50-95(B)']:.6f}\n\n")
            
            # 损失统计
            f.write("-" * 60 + "\n")
            f.write("Loss Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Final Train Box Loss:     {final['train/box_loss']:.6f}\n")
            f.write(f"Final Train Cls Loss:     {final['train/cls_loss']:.6f}\n")
            f.write(f"Final Train DFL Loss:     {final['train/dfl_loss']:.6f}\n")
            f.write(f"Final Val Box Loss:       {final['val/box_loss']:.6f}\n")
            f.write(f"Final Val Cls Loss:       {final['val/cls_loss']:.6f}\n")
            f.write(f"Final Val DFL Loss:       {final['val/dfl_loss']:.6f}\n\n")
            
            # 改进幅度
            initial = self.df.iloc[0]
            f.write("-" * 60 + "\n")
            f.write("Performance Improvement (First -> Last Epoch):\n")
            f.write("-" * 60 + "\n")
            f.write(f"Precision:       {initial['metrics/precision(B)']:.4f} -> {final['metrics/precision(B)']:.4f} (+{(final['metrics/precision(B)']-initial['metrics/precision(B)'])*100:.2f}%)\n")
            f.write(f"Recall:          {initial['metrics/recall(B)']:.4f} -> {final['metrics/recall(B)']:.4f} (+{(final['metrics/recall(B)']-initial['metrics/recall(B)'])*100:.2f}%)\n")
            f.write(f"mAP@0.5:         {initial['metrics/mAP50(B)']:.4f} -> {final['metrics/mAP50(B)']:.4f} (+{(final['metrics/mAP50(B)']-initial['metrics/mAP50(B)'])*100:.2f}%)\n")
            f.write(f"mAP@0.5:0.95:    {initial['metrics/mAP50-95(B)']:.4f} -> {final['metrics/mAP50-95(B)']:.4f} (+{(final['metrics/mAP50-95(B)']-initial['metrics/mAP50-95(B)'])*100:.2f}%)\n\n")
            
            f.write("=" * 60 + "\n")
            
        print(f"统计摘要已保存: {output_file}")
    
    def plot_confusion_matrix_from_yolo(self):
        """
        图8: 从YOLO混淆矩阵提取数据并重新绘制论文级可视化
        只保留 laying 和 standing 两个类别，去除 background
        """
        # 从YOLO官方混淆矩阵图片中提取的真实数据
        # 根据图片，提取 laying 和 standing 的混淆矩阵
        # laying (行0): 280个真实样本 - 280正确预测为laying, 7误判为standing
        # standing (行1): 1613个真实样本 - 10误判为laying, 1565正确预测为standing
        
        # 原始计数矩阵 (不包含background)
        # 行: 真实标签 [laying, standing]
        # 列: 预测标签 [laying, standing]
        cm_counts = np.array([
            [280, 7],      # laying: 280个被正确识别，7个被误判为standing
            [10, 1565]     # standing: 10个被误判为laying，1565个被正确识别
        ])
        
        # 归一化混淆矩阵（按行归一化，显示每个真实类别的预测分布）
        cm_normalized = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
        
        # 创建图表 - 只绘制归一化混淆矩阵
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制归一化混淆矩阵
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # 关闭网格
        ax.grid(False)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)
        
        # 设置刻度
        classes = ['Laying', 'Standing']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=18)
        ax.set_yticklabels(classes, fontsize=18)
        
        # 在每个格子中添加归一化数值
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                value = cm_normalized[i, j]
                ax.text(j, i, f'{value:.3f}',
                        ha="center", va="center",
                        color="white" if value > thresh else "black",
                        fontsize=22)
        
        # 设置标签
        ax.set_ylabel('True Label', fontsize=20, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=20, fontweight='bold')
        ax.set_title('Normalized Confusion Matrix', 
                     fontsize=22, fontweight='bold', pad=20)
        
        plt.tight_layout()
        self.save_figure(fig, '08_confusion_matrix_paper')
        plt.close()
        
        # 打印详细统计信息到控制台（不在图上显示）
        total_samples = cm_counts.sum()
        accuracy = np.trace(cm_counts) / total_samples
        
        # 计算每个类别的指标
        laying_precision = cm_counts[0, 0] / cm_counts[:, 0].sum()
        laying_recall = cm_counts[0, 0] / cm_counts[0, :].sum()
        laying_f1 = 2 * (laying_precision * laying_recall) / (laying_precision + laying_recall)
        
        standing_precision = cm_counts[1, 1] / cm_counts[:, 1].sum()
        standing_recall = cm_counts[1, 1] / cm_counts[1, :].sum()
        standing_f1 = 2 * (standing_precision * standing_recall) / (standing_precision + standing_recall)
        
        print("\n" + "="*60)
        print("混淆矩阵分析结果 (Laying vs Standing)")
        print("="*60)
        print(f"\n总样本数: {total_samples}")
        print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nLaying 类别:")
        print(f"  样本数: {cm_counts[0, :].sum()}")
        print(f"  Precision: {laying_precision:.4f}")
        print(f"  Recall: {laying_recall:.4f}")
        print(f"  F1-Score: {laying_f1:.4f}")
        print(f"\nStanding 类别:")
        print(f"  样本数: {cm_counts[1, :].sum()}")
        print(f"  Precision: {standing_precision:.4f}")
        print(f"  Recall: {standing_recall:.4f}")
        print(f"  F1-Score: {standing_f1:.4f}")
        print("="*60)
        
    def generate_all_plots(self):
        """生成所有图表"""
        print("\n开始生成训练结果可视化图表...")
        print("=" * 60)
        
        self.plot_loss_curves()
        self.plot_metrics_curves()
        self.plot_learning_rate()
        self.plot_comprehensive_comparison()
        self.plot_final_metrics_bar()
        self.plot_training_time()
        self.plot_convergence_analysis()
        self.plot_confusion_matrix_from_yolo()  # 使用新的方法
        self.generate_statistics_summary()
        
        print("\n" + "=" * 60)
        print("所有图表生成完成！")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


def main():
    """主函数"""
    # 设置CSV文件路径
    csv_path = "YOLOV11/output/yolov11s_cattle_20260109_223121_500/results.csv"
    # 创建可视化对象
    output_dir = "YOLOV11/output/yolov11s_cattle_20260109_223121_500/training_visualizations_results"    
    visualizer = TrainingResultsVisualizer(csv_path, output_dir)
    
    # 生成所有图表
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
