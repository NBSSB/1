# NEOGENESIS: LLM Evolutionary Strategies Experiment Framework

---

## 📋 Overview

NEOGENESIS is an experiment framework for studying "Emergence" and "Anti-fragility" effects in Large Language Models (LLMs).

**Core Question**: Can different prompting strategies enable LLM to show learning improvement across multiple Q&A rounds?

**Reference Paper**: "From Monolith to Archipelago: Emergence of Anti-Fragility in LLMs"

---

## 🧪 Experimental Results

### Accuracy Ranking (150 rounds)

| Rank | Strategy | Method | Avg Accuracy | Emergence |
|:---:|---------|--------|:---:|:---:|
| 🥇 | **v8_Island** | Memory Augmentation | **75%** | **+20%** ✅ |
| 🥈 | v11_Recur | Double Verification | 69% | 0% |
| 🥉 | v9_Meta | Chain-of-Thought | 62% | -8% |
| 4 | v6_Base | Direct Answer | 59% | +12% |
| 5 | v7_Evo | Self-Reflection | 53% | -8% |

### Key Findings

- ✅ **v8_Island succeeded**: Accuracy improved from 64% to 84% (+20%)
- ⚠️ Paper methods not fully reproducible: Traditional evolutionary strategies don't work well on MiniMax-M2.1
- 💡 Simple strategies more effective: Memory augmentation outperforms complex strategies

---

## 📁 File Structure

```
NEOGENESIS/
├── neogenesis.py                 # Original framework
├── neogenesis_lab.py            # ⭐ Main experiment (recommended)
├── neogenesis_reproduction.py    # Paper reproduction
├── neogenesis_roi_analysis.py   # ROI analysis
├── neogenesis_roi_analysis.png  # ROI chart
│
├── neogenesis_flashpoint_final.json   # Flashpoint results
├── neogenesis_iterative_final.json   # Iterative results
│
├── EXPERIMENT_SUMMARY.md        # Experiment report
├── README.md                    # Chinese version
└── README_EN.md                 # This file
```

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install requests numpy matplotlib
```

### Run Experiments

```bash
# Flashpoint mode (test different strategies)
python neogenesis_lab.py --mode flashpoint --rounds 50

# Iterative mode (same question repeated)
python neogenesis_lab.py --mode iterative --rounds 20
```

### Configuration

Edit the beginning of `neogenesis_lab.py`:

```python
API_KEY = "your-api-key-here"  # MiniMax API Key
API_HOST = "https://api.minimaxi.com/v1"
MODEL_NAME = "MiniMax-M2.1"
```

---

## 📖 Strategy Details

### v6_Base (Baseline)
- **Method**: Direct answer
- **Prompt**: `"Answer: {question}"`
- **Feature**: Simple and stable

### v7_Evo (Self-Reflection)
- **Method**: Answer twice, verify correctness
- **Feature**: Avoid obvious errors

### v8_Island ⭐ (Memory Augmentation)
- **Method**: Use previous round answer as hint
- **Prompt**: `Hint: {previous_answer}... Answer: {question}`
- **Feature**: Cross-round learning, best emergence

### v9_Meta (Chain-of-Thought)
- **Method**: Step-by-step reasoning
- **Feature**: Gradual reasoning

### v11_Recur (Double Verification)
- **Method**: Answer twice, confirm quality
- **Feature**: Most stable performance

---

## 🎯 Question Dataset

12 hard AI/ML questions that model often gets wrong:

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

## 📊 Data Analysis

### Run Analysis

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

### Output Format

JSON results contain:
- `history`: Scores per round for each strategy
- `fractal`: Detailed question-level data

---

## 🔬 Experiment Protocols

### Flashpoint Protocol
- Random question sampling from pool
- Test strategy performance across varying difficulties
- Suitable for quick evaluation

### Iterative Protocol
- Same question repeated multiple times
- Test strategy improvement on identical questions
- Suitable for studying learning effects

---

## ⚠️ FAQ

### Q: Why do advanced strategies perform worse?
A: 
1. High temperature increases randomness
2. Multi-step calls accumulate errors
3. MiniMax-M2.1 already has strong baseline capability

### Q: How to reproduce emergence?
A: Use v8_Island strategy - it achieves emergence through memory reuse

### Q: What if API calls fail?
A: Built-in 3x retry mechanism ensures stability

---

## 📝 Changelog

- **2026-02-19**: Completed 150 rounds, v8_Island shows +20% emergence
- **2026-02-18**: Fixed temperature issues, optimized strategies
- **2026-02-17**: Initial version, found questions too easy

---

## 🤝 Contributing

Issues and PRs welcome!

---

## 📄 License

MIT License

---

*Experiment completed on 2026-02-19*
