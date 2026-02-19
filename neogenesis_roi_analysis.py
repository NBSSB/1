#!/usr/bin/env python3
"""
NEOGENESIS ROI Analysis
==================
Verify "Intelligence Yield Curve" hypothesis:
- v6 (baseline): horizontal line
- v8/v11 (evolutionary): inverted U-curve
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from scipy import stats

def load_data():
    with open('neogenesis_flashpoint.json', 'r') as f:
        return json.load(f)

def calculate_roi(data, window=10):
    """计算滚动ROI"""
    n = len(data['v6_Base'])
    roi = {}
    
    # 修改成本模型:
    # v6: 成本=1 (基线不变)
    # v8: 成本=1+0.1*Round (降低增长系数,观察涌现)
    # v11: 成本=1+0.2*Round (降低增长系数)
    
    roi['v6'] = {
        'cost': [1] * n,
        'accuracy': data['v6_Base'],
        'yield': [a / 1 for a in data['v6_Base']]
    }
    
    roi['v8'] = {
        'cost': [1 + 0.1 * i for i in range(n)],
        'accuracy': data['v8_Island'],
        'yield': [data['v8_Island'][i] / (1 + 0.1 * i) for i in range(n)]
    }
    
    roi['v11'] = {
        'cost': [1 + 0.2 * i for i in range(n)],
        'accuracy': data['v11_Recur'],
        'yield': [data['v11_Recur'][i] / (1 + 0.2 * i) for i in range(n)]
    }
    
    return roi

def analyze_curve(yield_data, cost_data):
    """分析收益曲线特征"""
    n = len(yield_data)
    
    # 找到ROI峰值
    peak_idx = np.argmax(yield_data)
    peak_yield = yield_data[peak_idx]
    peak_cost = cost_data[peak_idx]
    
    # 计算曲线特征
    # 1. 前期斜率 (前1/3)
    early = yield_data[:n//3]
    early_slope = (early[-1] - early[0]) / len(early) if len(early) > 1 else 0
    
    # 2. 后期斜率 (后1/3) 
    late = yield_data[-n//3:]
    late_slope = (late[-1] - late[0]) / len(late) if len(late) > 1 else 0
    
    # 3. 曲线形态检验
    # 倒U型: 前期上升，后期下降
    is_inverted_u = early_slope > 0 and late_slope < 0
    
    return {
        'peak_idx': peak_idx,
        'peak_yield': peak_yield,
        'peak_cost': peak_cost,
        'early_slope': early_slope,
        'late_slope': late_slope,
        'is_inverted_u': is_inverted_u
    }

def main():
    data = load_data()
    n = len(data['v6_Base'])
    print(f"=" * 70)
    print(f"NEOGENESIS ROI Analysis Report ({n} rounds)")
    print(f"=" * 70)
    
    roi = calculate_roi(data)
    
    # Analyze each strategy
    print("\n[ROI Curve Analysis]")
    print("-" * 70)
    
    results = {}
    for strategy in ['v6', 'v8', 'v11']:
        analysis = analyze_curve(roi[strategy]['yield'], roi[strategy]['cost'])
        results[strategy] = analysis
        
        print(f"\n{strategy}:")
        print(f"  Peak ROI: {analysis['peak_yield']:.4f} @ Round {analysis['peak_idx']}")
        print(f"  Early slope: {analysis['early_slope']:.4f}")
        print(f"  Late slope: {analysis['late_slope']:.4f}")
        print(f"  Inverted U: {'YES' if analysis['is_inverted_u'] else 'NO'}")
    
    # 绘制ROI曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 准确率曲线
    ax1 = axes[0, 0]
    rounds = range(1, n+1)
    ax1.plot(rounds, data['v6_Base'], label='v6_Base', alpha=0.7)
    ax1.plot(rounds, data['v8_Island'], label='v8_Island', alpha=0.7)
    ax1.plot(rounds, data['v11_Recur'], label='v11_Recur', alpha=0.7)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROI曲线
    ax2 = axes[0, 1]
    ax2.plot(rounds, roi['v6']['yield'], label='v6 (Cost=1)', alpha=0.7)
    ax2.plot(rounds, roi['v8']['yield'], label='v8 (Cost=1+0.1*Round)', alpha=0.7)
    ax2.plot(rounds, roi['v11']['yield'], label='v11 (Cost=1+0.2*Round)', alpha=0.7)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('ROI (Accuracy/Cost)')
    ax2.set_title('ROI (Intelligence Yield Curve)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 标记峰值
    for strategy, color in [('v6', 'blue'), ('v8', 'orange'), ('v11', 'green')]:
        peak_idx = results[strategy]['peak_idx']
        peak_yield = results[strategy]['peak_yield']
        ax2.scatter([peak_idx+1], [peak_yield], color=color, s=100, zorder=5, marker='*')
    
    # 3. 成本曲线
    ax3 = axes[1, 0]
    ax3.plot(rounds, roi['v6']['cost'], label='v6', linestyle='--')
    ax3.plot(rounds, roi['v8']['cost'], label='v8', linestyle='--')
    ax3.plot(rounds, roi['v11']['cost'], label='v11', linestyle='--')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cost')
    ax3.set_title('Cost Growth')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积收益
    ax4 = axes[1, 1]
    ax4.plot(rounds, np.cumsum(data['v6_Base']), label='v6_Base')
    ax4.plot(rounds, np.cumsum(data['v8_Island']), label='v8_Island')
    ax4.plot(rounds, np.cumsum(data['v11_Recur']), label='v11_Recur')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Cumulative Accuracy')
    ax4.set_title('Cumulative Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neogenesis_roi_analysis.png', dpi=150)
    print(f"\n[图表已保存: neogenesis_roi_analysis.png]")
    
    # Hypothesis verification
    print("\n" + "=" * 70)
    print("[Hypothesis Verification: Intelligence Yield Curve]")
    print("=" * 70)
    
    print("\n1. v6 (baseline) - horizontal line hypothesis:")
    v6_slope, _, r, p, _ = stats.linregress(range(n), roi['v6']['yield'])
    print(f"   Slope: {v6_slope:.6f} (near 0?) {'PASS' if abs(v6_slope) < 0.01 else 'FAIL'}")
    print(f"   R-squared: {r**2:.4f}")
    
    print("\n2. v8 (evolutionary) - inverted U-curve hypothesis:")
    print(f"   Early rising: {'PASS' if results['v8']['early_slope'] > 0 else 'FAIL'} (slope={results['v8']['early_slope']:.4f})")
    print(f"   Late falling: {'PASS' if results['v8']['late_slope'] < 0 else 'FAIL'} (slope={results['v8']['late_slope']:.4f})")
    print(f"   Peak round: {results['v8']['peak_idx']}")
    
    print("\n3. v11 (recursive) - inverted U-curve hypothesis:")
    print(f"   Early rising: {'PASS' if results['v11']['early_slope'] > 0 else 'FAIL'} (slope={results['v11']['early_slope']:.4f})")
    print(f"   Late falling: {'PASS' if results['v11']['late_slope'] < 0 else 'FAIL'} (slope={results['v11']['late_slope']:.4f})")
    print(f"   Peak round: {results['v11']['peak_idx']}")
    
    # Optimal iteration count
    print("\n" + "=" * 70)
    print("[Optimal Iteration Count (Flashpoint)]")
    print("=" * 70)
    print(f"  v8 optimal: Round {results['v8']['peak_idx']} (ROI={results['v8']['peak_yield']:.4f})")
    print(f"  v11 optimal: Round {results['v11']['peak_idx']} (ROI={results['v11']['peak_yield']:.4f})")
    
    # Final conclusion
    print("\n" + "=" * 70)
    print("[Conclusion]")
    print("=" * 70)
    
    # Compare final ROI vs initial ROI
    v8_initial = roi['v8']['yield'][0]
    v8_final = roi['v8']['yield'][-1]
    v11_initial = roi['v11']['yield'][0]
    v11_final = roi['v11']['yield'][-1]
    
    print(f"v8 ROI change: {v8_initial:.4f} -> {v8_final:.4f} ({'+' if v8_final > v8_initial else ''}{(v8_final-v8_initial)*100:.1f}%)")
    print(f"v11 ROI change: {v11_initial:.4f} -> {v11_final:.4f} ({'+' if v11_final > v11_initial else ''}{(v11_final-v11_initial)*100:.1f}%)")
    
    if v8_final < v8_initial and v11_final < v11_initial:
        print("\n[PASS] Hypothesis verified: Evolutionary strategies show diminishing marginal returns")
    elif v8_final > v8_initial or v11_final > v11_initial:
        print("\n[PARTIAL] Hypothesis partially verified: Marginal returns not clearly diminishing")

if __name__ == "__main__":
    main()
