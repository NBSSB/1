# NEOGENESIS: LLM进化策略实验框架

[English](README_EN.md) | [中文](README.md)

---

## 📋 项目简介

NEOGENESIS是一个用于研究LLM(大语言模型)"涌现"(Emergence)和"反脆弱"(Anti-fragility)效应的实验框架。

**核心问题**: 不同的提示策略是否能让LLM在多轮问答中表现出学习改进？

**论文参考**: "From Monolith to Archipelago: Emergence of Anti-Fragility in LLMs"

---

## 🧪 实验结果

### 准确率排名 (150轮测试)

| 排名 | 策略 | 方法 | 平均准确率 | 涌现效应 |
|:---:|------|------|:---:|:---:|
| 🥇 | **v8_Island** | 记忆增强 | **75%** | **+20%** ✅ |
| 🥈 | v11_Recur | 双重验证 | 69% | 0% |
| 🥉 | v9_Meta | 思维链 | 62% | -8% |
| 4 | v6_Base | 直接回答 | 59% | +12% |
| 5 | v7_Evo | 自省 | 53% | -8% |

### 关键发现

- ✅ **v8_Island成功涌现**: 准确率从64%提升到84% (+20%)
- ⚠️ 论文方法无法完全复现: 传统进化策略在MiniMax-M2.1上效果不佳
- 💡 简单策略(记忆增强)比复杂策略更有效

---

## 📁 文件结构

```
NEOGENESIS/
├── neogenesis.py                 # 原始框架 (v6, v8, v9, v11)
├── neogenesis_lab.py            # ⭐ 主实验程序 (推荐)
├── neogenesis_reproduction.py   # 论文复现实验
├── neogenesis_roi_analysis.py    # ROI收益分析
├── neogenesis_roi_analysis.png   # ROI图表
│
├── neogenesis_flashpoint_final.json   # Flashpoint模式结果
├── neogenesis_iterative_final.json     # Iterative模式结果
│
├── EXPERIMENT_SUMMARY.md         # 实验总结报告
├── README.md                    # 本文档 (中文)
└── README_EN.md                 # English version
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install requests numpy matplotlib
```

### 运行实验

```bash
# Flashpoint模式 (快速测试不同策略)
python neogenesis_lab.py --mode flashpoint --rounds 50

# Iterative模式 (测试同一问题重复问答)
python neogenesis_lab.py --mode iterative --rounds 20
```

### 配置

编辑 `neogenesis_lab.py` 开头部分:

```python
API_KEY = "your-api-key-here"  # MiniMax API Key
API_HOST = "https://api.minimaxi.com/v1"
MODEL_NAME = "MiniMax-M2.1"
```

---

## 📖 策略详解

### v6_Base (基线)
- **方法**: 直接回答
- **提示**: `"Answer: {问题}"`
- **特点**: 简单稳定，作为基准

### v7_Evo (自省)
- **方法**: 答两次，验证正确则返回
- **提示**: 首次回答 + 验证确认
- **特点**: 避免明显错误

### v8_Island ⭐ (记忆增强)
- **方法**: 用前一轮答案作为提示
- **提示**: `Hint: {之前答案}... Answer: {问题}`
- **特点**: 实现跨轮次学习，涌现效果最佳

### v9_Meta (思维链)
- **方法**: 分步思考
- **提示**: `"Think: {问题}"` → `"Based on above: {问题}"`
- **特点**: 逐步推理

### v11_Recur (双重验证)
- **方法**: 答两次，确认答案质量
- **提示**: 快速回答 + 确认
- **特点**: 最稳定的表现

---

## 🎯 题库说明

### HARD_EXPERIMENT_DATASET (12题)
超难AI/ML专业问题，模型容易出错:

```python
[
    {"q": "What is the exact beta1 in Adam optimizer?", "a": ["0.9"]},
    {"q": "What is the exact beta2 in Adam optimizer?", "a": ["0.999"]},
    {"q": "What is the exact epsilon in Adam optimizer?", "a": ["1e-8"]},
    {"q": "What is the exact dropout in T5-base?", "a": ["0.1"]},
    {"q": "What is the exact warmup steps in original T5?", "a": ["1000"]},
    {"q": "What is the exact hidden size in ALBERT-xxlarge?", "a": ["4096"]},
    {"q": "What is the exact attention heads in ALBERT-xxlarge?", "a": ["64"]},
    {"q": "What is the exact vocab size of GPT-2?", "a": ["50257"]},
    {"q": "What is the exact training steps in original BERT?", "a": ["1000000", "1e6"]},
    {"q": "What is the exact batch size in BERT pretraining?", "a": ["256"]},
    {"q": "What is the exact learning rate in T5-small?", "a": ["1e-3", "0.001"]},
    {"q": "What is the FFN dimension ratio in T5-base?", "a": ["4"]},
]
```

---

## 📊 数据分析

### 运行分析脚本

```bash
python -c "
import json
with open('neogenesis_flashpoint_final.json') as f:
    data = json.load(f)

for strat, scores in data['history'].items():
    acc = sum(scores) / len(scores) * 100
    print(f'{strat}: {acc:.1f}%')
"
```

### 输出格式

JSON结果包含:
- `history`: 每轮各策略得分
- `fractal`: 详细题目级别数据

---

## 🔬 实验协议

### Flashpoint Protocol
- 随机从题库抽取题目
- 测试策略在随机难度下的表现
- 适合快速评估

### Iterative Protocol
- 同一问题重复测试多次
- 测试策略在同一问题上的改进
- 适合研究学习效应

---

## ⚠️ 常见问题

### Q: 为什么高级策略效果反而差?
A: 
1. Temperature过高会导致随机性增加
2. 多步骤调用会累积误差
3. MiniMax-M2.1基础能力已很强

### Q: 如何复现涌现?
A: 使用v8_Island策略，它通过记忆复用实现跨轮次学习

### Q: API调用失败怎么办?
A: 程序内置3次重试机制，确保稳定性

---

## 📝 实验日志

- **2026-02-19**: 完成150轮测试，v8_Island涌现+20%
- **2026-02-18**: 修复temperature问题，优化策略
- **2026-02-17**: 初始版本，发现题目太简单

---

## 🤝 贡献

欢迎提交Issue和PR！

---

## 📄 License

MIT License

---

*实验完成于 2026-02-19*
