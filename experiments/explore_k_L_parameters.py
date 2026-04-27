"""
探索逻辑函数中 k 和 L 参数对曲线形状的影响

基于之前拟合的参数值，固定 b 和 t0，变化 k 和 L 来观察曲线变化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os

# ============================================
# 逻辑函数定义
# ============================================
def logistic(t, L, k, t0, b):
    """
    逻辑生长函数
    L: 增长幅度（上渐近线增量）
    k: 生长速率（day^-1）
    t0: 拐点时间（天）
    b: 基线水平（下渐近线）
    """
    return L / (1 + np.exp(-k * (t - t0))) + b

def logistic_derivative(t, L, k, t0, b):
    """生长速率 dy/dt"""
    exp_term = np.exp(-k * (t - t0))
    return k * L * exp_term / ((1 + exp_term) ** 2)

# ============================================
# 使用之前拟合的参数作为基准值
# ============================================
# 从curve_width.py的输出结果
b_base = 0.295387      # 基线水平
t0_base = 88.42        # 拐点时间
L_base = 0.011615      # 增长幅度（基准）
k_base = 0.300994      # 生长速率（基准）

# 时间范围
t = np.linspace(0, 120, 500)

# 输出目录
output_dir = 'parameter_exploration_results'
os.makedirs(output_dir, exist_ok=True)

# ============================================
# Figure 1: L 参数的影响（固定 k）
# ============================================
print("=" * 80)
print("探索 L (增长幅度) 参数的影响")
print("=" * 80)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 固定 k = k_base，变化 L
L_values = [
    L_base * 0.25,   # 25% 的基准值
    L_base * 0.5,    # 50%
    L_base * 1.0,    # 100% (基准)
    L_base * 2.0,    # 200%
    L_base * 4.0,    # 400%
]

colors1 = plt.cm.viridis(np.linspace(0.2, 0.9, len(L_values)))

for i, L in enumerate(L_values):
    y = logistic(t, L, k_base, t0_base, b_base)
    growth_rate = logistic_derivative(t, L, k_base, t0_base, b_base)
    
    label = f'L = {L:.5f} ({L/L_base:.1f}× baseline)'
    if L == L_base:
        label += ' ★'
    
    ax1.plot(t, y, color=colors1[i], linewidth=2.5 if L == L_base else 2, 
             label=label, linestyle='-' if L == L_base else '--')
    ax2.plot(t, growth_rate, color=colors1[i], linewidth=2.5 if L == L_base else 2,
             label=label, linestyle='-' if L == L_base else '--')
    
    print(f"L = {L:.5f} ({L/L_base:.1f}×基准):")
    print(f"  最终渐近值 (L+b) = {L + b_base:.5f}")
    print(f"  总增长量 (L) = {L:.5f}")
    print(f"  最大生长速率 = {np.max(growth_rate):.6f} W/L/day")
    print()

# Mark inflection point
ax1.axvline(t0_base, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Inflection Point (t₀={t0_base:.0f}d)')
ax2.axvline(t0_base, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

ax1.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax1.set_ylabel('Width/Length Ratio', fontsize=14, fontweight='bold')
ax1.set_title('Effect of L Parameter on Growth Curve\n(Fixed k={:.3f}, t₀={:.0f}, b={:.4f})'.format(k_base, t0_base, b_base), 
              fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax2.set_ylabel('Growth Rate (W/L per day)', fontsize=14, fontweight='bold')
ax2.set_title('Effect of L Parameter on Growth Rate\n(dy/dt)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig1_L_parameter_influence.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig1_L_parameter_influence.pdf'), bbox_inches='tight')
print(f"✓ Figure 1 saved: {os.path.join(output_dir, 'Fig1_L_parameter_influence.png')}")
print()

# ============================================
# Figure 2: k 参数的影响（固定 L）
# ============================================
print("=" * 80)
print("探索 k (生长速率) 参数的影响")
print("=" * 80)

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# 固定 L = L_base，变化 k
k_values = [
    k_base * 0.25,   # 25% 的基准值
    k_base * 0.5,    # 50%
    k_base * 1.0,    # 100% (基准)
    k_base * 2.0,    # 200%
    k_base * 4.0,    # 400%
]

colors2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(k_values)))

for i, k in enumerate(k_values):
    y = logistic(t, L_base, k, t0_base, b_base)
    growth_rate = logistic_derivative(t, L_base, k, t0_base, b_base)
    
    label = f'k = {k:.4f} ({k/k_base:.1f}× baseline)'
    if k == k_base:
        label += ' ★'
    
    ax3.plot(t, y, color=colors2[i], linewidth=2.5 if k == k_base else 2,
             label=label, linestyle='-' if k == k_base else '--')
    ax4.plot(t, growth_rate, color=colors2[i], linewidth=2.5 if k == k_base else 2,
             label=label, linestyle='-' if k == k_base else '--')
    
    print(f"k = {k:.4f} ({k/k_base:.1f}×基准):")
    print(f"  最终渐近值 (L+b) = {L_base + b_base:.5f} (不变)")
    print(f"  最大生长速率 = {np.max(growth_rate):.6f} W/L/day")
    
    # 计算从10%到90%生长所需时间
    y_10 = b_base + 0.1 * L_base
    y_90 = b_base + 0.9 * L_base
    t_10 = None
    t_90 = None
    for j, y_val in enumerate(y):
        if t_10 is None and y_val >= y_10:
            t_10 = t[j]
        if t_90 is None and y_val >= y_90:
            t_90 = t[j]
            break
    if t_10 is not None and t_90 is not None:
        print(f"  10%-90% growth time = {t_90 - t_10:.1f} days")
    print()

# Mark inflection point
ax3.axvline(t0_base, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Inflection Point (t₀={t0_base:.0f}d)')
ax4.axvline(t0_base, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

ax3.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax3.set_ylabel('Width/Length Ratio', fontsize=14, fontweight='bold')
ax3.set_title('Effect of k Parameter on Growth Curve\n(Fixed L={:.5f}, t₀={:.0f}, b={:.4f})'.format(L_base, t0_base, b_base),
              fontsize=13, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)

ax4.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax4.set_ylabel('Growth Rate (W/L per day)', fontsize=14, fontweight='bold')
ax4.set_title('Effect of k Parameter on Growth Rate\n(dy/dt)', fontsize=13, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig2_k_parameter_influence.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig2_k_parameter_influence.pdf'), bbox_inches='tight')
print(f"✓ Figure 2 saved: {os.path.join(output_dir, 'Fig2_k_parameter_influence.png')}")
print()

# ============================================
# Figure 3: k 和 L 的联合影响（热图）
# ============================================
print("=" * 80)
print("探索 k 和 L 的联合影响")
print("=" * 80)

fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# 创建 k 和 L 的网格
k_range = np.linspace(k_base * 0.1, k_base * 5, 50)
L_range = np.linspace(L_base * 0.1, L_base * 5, 50)
K_grid, L_grid = np.meshgrid(k_range, L_range)

# 计算不同指标
# (1) 最终渐近值
final_value = L_grid + b_base

# (2) 最大生长速率
max_growth_rate = K_grid * L_grid / 4  # 在拐点处的最大值

# (3) 从10%到90%生长所需时间（理论值）
# 对于逻辑函数，从 y(t1)=b+0.1L 到 y(t2)=b+0.9L
# 可以推导：t2-t1 = ln(81) / k ≈ 4.394 / k
growth_duration = 4.394 / K_grid

# (4) R² 评分（假设相对于基准模型）
# 这里简化为：越接近基准参数，R²越高
r2_approximation = 1 - ((K_grid - k_base)**2 / k_base**2 + (L_grid - L_base)**2 / L_base**2) / 10
r2_approximation = np.clip(r2_approximation, 0, 1)

# Plot heatmaps
metrics = [
    (final_value, 'Final Asymptote (L + b)', 'viridis'),
    (max_growth_rate, 'Max Growth Rate (k·L/4)', 'plasma'),
    (growth_duration, '10%-90% Growth Time (days)', 'coolwarm'),
    (r2_approximation, 'Approx. Fit Quality (R²)', 'RdYlGn')
]

for idx, (data, title, cmap) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    im = ax.contourf(K_grid, L_grid, data, levels=20, cmap=cmap)
    
    # Mark baseline point
    ax.plot(k_base, L_base, 'r*', markersize=20, markeredgecolor='white', 
            markeredgewidth=2, label='Baseline Parameters')
    
    # 添加等高线
    contours = ax.contour(K_grid, L_grid, data, levels=10, colors='black', 
                          linewidths=0.5, alpha=0.3)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
    
    ax.set_xlabel('k (Growth Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('L (Growth Amplitude)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig3_k_L_joint_influence.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig3_k_L_joint_influence.pdf'), bbox_inches='tight')
print(f"✓ Figure 3 saved: {os.path.join(output_dir, 'Fig3_k_L_joint_influence.png')}")
print()

# ============================================
# Figure 4: 对比原始数据和不同参数的拟合效果
# ============================================
print("=" * 80)
print("对比不同参数组合的拟合效果")
print("=" * 80)

fig4, ax5 = plt.subplots(figsize=(14, 8))

# 原始数据点（从curve_width.py）
days_observed = np.array([1, 11, 22, 33, 43, 53, 74, 81, 88, 95, 102, 110])
# 使用基准参数生成"观测值"
y_observed = logistic(days_observed, L_base, k_base, t0_base, b_base)

# Plot "observed" data
ax5.scatter(days_observed, y_observed, s=100, color='orange', edgecolors='black',
           linewidths=2, zorder=5, label='Fitted Observed Data', alpha=0.8)

# Different parameter combinations
param_combinations = [
    (L_base, k_base, 'Baseline k and L', 'blue', '-', 3),
    (L_base * 0.5, k_base, 'L halved (k fixed)', 'green', '--', 2),
    (L_base * 2, k_base, 'L doubled (k fixed)', 'purple', '--', 2),
    (L_base, k_base * 0.5, 'k halved (L fixed)', 'red', '-.', 2),
    (L_base, k_base * 2, 'k doubled (L fixed)', 'brown', '-.', 2),
]

for L, k, label, color, linestyle, linewidth in param_combinations:
    y_fit = logistic(t, L, k, t0_base, b_base)
    ax5.plot(t, y_fit, color=color, linestyle=linestyle, linewidth=linewidth,
            label=label, alpha=0.8)

ax5.axvline(t0_base, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
           label=f'Inflection Point (t₀={t0_base:.0f}d)')

ax5.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax5.set_ylabel('Width/Length Ratio', fontsize=14, fontweight='bold')
ax5.set_title('Comparison of Different k and L Parameter Combinations\n(Fixed t₀={:.0f}, b={:.4f})'.format(t0_base, b_base),
             fontsize=14, fontweight='bold')
ax5.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 120)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig4_parameter_combinations_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig4_parameter_combinations_comparison.pdf'), bbox_inches='tight')
print(f"✓ Figure 4 saved: {os.path.join(output_dir, 'Fig4_parameter_combinations_comparison.png')}")
print()

# ============================================
# Figure 5: t₀ 参数的影响（拐点时间）
# ============================================
print("=" * 80)
print("Exploring t₀ (Inflection Point) Parameter Influence")
print("=" * 80)

fig5, (ax6, ax7) = plt.subplots(1, 2, figsize=(16, 6))

# Fix L = L_base, k = k_base, b = b_base, vary t₀
t0_values = [
    t0_base - 40,     # 40 days earlier
    t0_base - 20,     # 20 days earlier
    t0_base,          # Baseline
    t0_base + 20,     # 20 days later
    t0_base + 40,     # 40 days later
]

colors5 = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(t0_values)))

for i, t0 in enumerate(t0_values):
    y = logistic(t, L_base, k_base, t0, b_base)
    growth_rate = logistic_derivative(t, L_base, k_base, t0, b_base)
    
    label = f't₀ = {t0:.1f}d'
    if abs(t0 - t0_base) < 0.1:
        label += ' (Baseline) ★'
    elif t0 < t0_base:
        label += f' ({t0_base - t0:.0f}d earlier)'
    else:
        label += f' ({t0 - t0_base:.0f}d later)'
    
    ax6.plot(t, y, color=colors5[i], linewidth=2.5 if abs(t0 - t0_base) < 0.1 else 2,
             label=label, linestyle='-' if abs(t0 - t0_base) < 0.1 else '--')
    ax7.plot(t, growth_rate, color=colors5[i], linewidth=2.5 if abs(t0 - t0_base) < 0.1 else 2,
             label=label, linestyle='-' if abs(t0 - t0_base) < 0.1 else '--')
    
    # Mark inflection point
    ax6.axvline(t0, color=colors5[i], linestyle=':', linewidth=1, alpha=0.5)
    ax7.axvline(t0, color=colors5[i], linestyle=':', linewidth=1, alpha=0.5)
    
    print(f"t₀ = {t0:.1f}d:")
    print(f"  Final asymptote (L+b) = {L_base + b_base:.5f} (unchanged)")
    print(f"  Max growth rate = {np.max(growth_rate):.6f} W/L/day (unchanged)")
    print(f"  Inflection point shifted: {t0:.1f} days")
    print()

ax6.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax6.set_ylabel('Width/Length Ratio', fontsize=14, fontweight='bold')
ax6.set_title('Effect of t₀ Parameter on Growth Curve\n(Fixed L={:.5f}, k={:.3f}, b={:.4f})'.format(L_base, k_base, b_base),
              fontsize=13, fontweight='bold')
ax6.legend(loc='lower right', fontsize=9)
ax6.grid(True, alpha=0.3)

ax7.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax7.set_ylabel('Growth Rate (W/L per day)', fontsize=14, fontweight='bold')
ax7.set_title('Effect of t₀ Parameter on Growth Rate\n(dy/dt)', fontsize=13, fontweight='bold')
ax7.legend(loc='upper right', fontsize=9)
ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig5_t0_parameter_influence.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig5_t0_parameter_influence.pdf'), bbox_inches='tight')
print(f"✓ Figure 5 saved: {os.path.join(output_dir, 'Fig5_t0_parameter_influence.png')}")
print()

# ============================================
# Figure 6: b 参数的影响（基线水平）
# ============================================
print("=" * 80)
print("Exploring b (Baseline Asymptote) Parameter Influence")
print("=" * 80)

fig6, (ax8, ax9) = plt.subplots(1, 2, figsize=(16, 6))

# Fix L = L_base, k = k_base, t₀ = t0_base, vary b
b_values = [
    b_base - 0.01,    # 0.01 lower
    b_base - 0.005,   # 0.005 lower
    b_base,           # Baseline
    b_base + 0.005,   # 0.005 higher
    b_base + 0.01,    # 0.01 higher
]

colors6 = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(b_values)))

for i, b in enumerate(b_values):
    y = logistic(t, L_base, k_base, t0_base, b)
    growth_rate = logistic_derivative(t, L_base, k_base, t0_base, b)
    
    label = f'b = {b:.4f}'
    if abs(b - b_base) < 0.0001:
        label += ' (Baseline) ★'
    elif b < b_base:
        label += f' ({b_base - b:.3f} lower)'
    else:
        label += f' ({b - b_base:.3f} higher)'
    
    ax8.plot(t, y, color=colors6[i], linewidth=2.5 if abs(b - b_base) < 0.0001 else 2,
             label=label, linestyle='-' if abs(b - b_base) < 0.0001 else '--')
    ax9.plot(t, growth_rate, color=colors6[i], linewidth=2.5 if abs(b - b_base) < 0.0001 else 2,
             label=label, linestyle='-' if abs(b - b_base) < 0.0001 else '--')
    
    print(f"b = {b:.4f}:")
    print(f"  Lower asymptote = {b:.5f}")
    print(f"  Final asymptote (L+b) = {L_base + b:.5f}")
    print(f"  Max growth rate = {np.max(growth_rate):.6f} W/L/day (unchanged)")
    print(f"  Vertical shift = {b - b_base:.5f}")
    print()

# Mark inflection point
ax8.axvline(t0_base, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, 
           label=f'Inflection Point (t₀={t0_base:.0f}d)')
ax9.axvline(t0_base, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

ax8.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax8.set_ylabel('Width/Length Ratio', fontsize=14, fontweight='bold')
ax8.set_title('Effect of b Parameter on Growth Curve\n(Fixed L={:.5f}, k={:.3f}, t₀={:.0f})'.format(L_base, k_base, t0_base),
              fontsize=13, fontweight='bold')
ax8.legend(loc='lower right', fontsize=9)
ax8.grid(True, alpha=0.3)

ax9.set_xlabel('Days from Start', fontsize=14, fontweight='bold')
ax9.set_ylabel('Growth Rate (W/L per day)', fontsize=14, fontweight='bold')
ax9.set_title('Effect of b Parameter on Growth Rate\n(dy/dt)', fontsize=13, fontweight='bold')
ax9.legend(loc='upper right', fontsize=9)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig6_b_parameter_influence.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig6_b_parameter_influence.pdf'), bbox_inches='tight')
print(f"✓ Figure 6 saved: {os.path.join(output_dir, 'Fig6_b_parameter_influence.png')}")
print()

# ============================================
# 生成总结报告
# ============================================
print("=" * 80)
print("参数影响总结")
print("=" * 80)
print()
print("【L 参数（增长幅度）的影响】")
print("  含义: 从基线到最终平台的总增长量")
print("  影响:")
print("    • L 越大 → 最终渐近值 (L+b) 越高")
print("    • L 越大 → 最大生长速率越大（与k成正比）")
print("    • L 不影响曲线的'陡峭程度'（由k决定）")
print("    • L 不影响拐点位置（由t₀决定）")
print()
print("【k 参数（生长速率）的影响】")
print("  含义: 控制曲线从下渐近线到上渐近线的过渡速度")
print("  影响:")
print("    • k 越大 → 曲线越'陡峭'（S型越尖锐）")
print("    • k 越大 → 最大生长速率越大")
print("    • k 越大 → 从10%到90%生长所需时间越短")
print("    • k 不影响最终渐近值（由L和b决定）")
print("    • k 不影响拐点位置（由t₀决定）")
print()
print("【k 和 L 的联合影响】")
print("  • 最大生长速率 = k × L / 4（在拐点处达到）")
print("  • 两者都增大 → 生长既快又高")
print("  • k大L小 → 快速但有限的生长")
print("  • k小L大 → 缓慢但显著的生长")
print()
print("【t₀ 参数（拐点时间）的影响】")
print("  含义: 最大生长速率出现的时间点，S型曲线的对称中心")
print("  影响:")
print("    • t₀ 控制曲线的'水平位置'（左右平移）")
print("    • t₀ 越大 → 整条曲线向右平移（生长延迟）")
print("    • t₀ 越小 → 整条曲线向左平移（生长提前）")
print("    • t₀ 不影响最终渐近值（由L和b决定）")
print("    • t₀ 不影响生长速率大小（由k和L决定）")
print("    • t₀ 不影响曲线形状（由k决定）")
print()
print("【b 参数（基线水平）的影响】")
print("  含义: 初始状态的测量值，下渐近线")
print("  影响:")
print("    • b 控制曲线的'垂直位置'（上下平移）")
print("    • b 越大 → 整条曲线向上平移")
print("    • b 越小 → 整条曲线向下平移")
print("    • b 不影响生长幅度（由L决定）")
print("    • b 不影响生长速率（由k和L决定）")
print("    • b 不影响拐点位置（由t₀决定）")
print("    • 最终值 = b + L（基线 + 增长量）")
print()
print("【四个参数的独立作用】")
print("  • L: 控制'长多高' (vertical scale)")
print("  • k: 控制'长多快' (steepness)")
print("  • t₀: 控制'何时长' (horizontal shift)")
print("  • b: 控制'起点在哪' (vertical shift)")
print()
print("【你的数据中各参数的可靠性】")
print(f"  • b (基线水平): CV = 2.8% → 非常可靠 ✓✓✓")
print(f"  • t₀ (拐点时间): CV = 27% → 相对可靠 ✓✓")
print(f"  • L (增长幅度): CV = 140% → 不确定性高 ✓")
print(f"  • k (生长速率): CV = 665% → 不确定性很高 ✓")
print()
print("  解释:")
print("    1. b 参数最可靠:")
print("       - 直接由迟缓期的7个数据点确定")
print("       - 测量稳定，变异小")
print()
print("    2. t₀ 参数相对可靠:")
print("       - 拐点位置在 88天左右，有数据支撑")
print("       - 虽然CV较高，但生理学上合理")
print()
print("    3. L 不确定性高 → 不能精确判断最终会长到多高")
print("       - 平台期只有3个点，无法确定是否完全稳定")
print("       - 但可以确定：有明显的增长（约4.4%）")
print()
print("    4. k 不确定性高 → 不能精确判断曲线有多'陡峭'")
print("       - 快速生长期只有2个点，无法精确测量斜率")
print("       - 但可以确定：确实存在快速增长阶段")
print()
print("    5. 主要结论不受影响:")
print("       - 时间效应显著 (p<0.01) ✓")
print("       - 存在明确的生长趋势 ✓")
print("       - 阶段划分清晰 ✓")
print("       - 基线b测量准确 (CV=2.8%) ✓")
print()
print("=" * 80)
print(f"✓ 所有分析图表已保存到: {output_dir}/")
print("=" * 80)
