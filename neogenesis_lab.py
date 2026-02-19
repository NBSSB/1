#!/usr/bin/env python3
"""
NEOGENESIS LAB: THE UNIFIED RESEARCH ENGINE
===========================================
Integrates:
- Strategies: v6(Base), v7(Evo), v8(Island), v9(Router), v11(Oroboros)
- Datasets: Logic (System 2), Riddle (System 1), Complex (System 3)
- Protocols: Flashpoint (ROI), Eternity (Stability)
"""

import json
import time
import random
import math
import os
import argparse
import concurrent.futures
from datetime import datetime

# --- 1. CONFIGURATION ---
API_KEY = "sk-cp-MobgKZjVQlhQydDz1sbtjitU8lkP5E00cMsbYLs_9d6ykhL-k_XvRcuDp8TZTNx63L8I--l0fT_oJ4css0cttKbeeJ0d1hIAm48_FeyMHUV5CnIHWNJe_BA"
API_HOST = "https://api.minimaxi.com/v1"
MODEL_NAME = "MiniMax-M2.1"
MAX_WORKERS = 20

# --- 2. TIER-X DATASET (全地形题库 - 包含专业领域问题) ---
DATASET = {
    "Logic": [ # System 2: 深度推理
        {"q": "Prove that sqrt(2) is irrational.", "a": ["contradiction", "a/b"]},
        {"q": "Solve T(n) = 2T(n/2) + n.", "a": ["n log n"]},
        {"q": "Find GCD of 144 and 96.", "a": ["48"]},
        {"q": "Explain P vs NP.", "a": ["polynomial"]},
        {"q": "Calculate derivative of x^3 * sin(x).", "a": ["3x^2", "cos"]},
        {"q": "Find eigenvalues of [[1, 2], [2, 1]].", "a": ["3", "-1"]},
        {"q": "Convert 100 binary to decimal.", "a": ["4"]},
        {"q": "What is the probability of 3 heads in 5 flips?", "a": ["0.3125"]},
        {"q": "Solve x^3 - 6x^2 + 11x - 6 = 0.", "a": ["1", "2", "3"]},
        {"q": "Derive Schwarzschild radius.", "a": ["2gm/c^2"]},
    ],
    # 专业知识盲区 - 模型容易混淆
    "Niche": [
        {"q": "What is the ISO 3166-1 alpha-3 code for North Korea?", "a": ["PRK"]},
        {"q": "Who was the first person to win the Turing Award?", "a": ["Perlis"]},
        {"q": "What year was the first Emmy Award given?", "a": ["1949"]},
        {"q": "What is the boiling point of Liquid Nitrogen in Celsius?", "a": ["-196"]},
        {"q": "What is the atomic number of Americium?", "a": ["95"]},
        {"q": "Who wrote 'The Art of Computer Programming Vol 1' first edition ISBN?", "a": ["0201896831"]},
        {"q": "What is the MIME type for XML?", "a": ["application/xml"]},
        {"q": "What year did the Great Schism in Christianity occur?", "a": ["1054"]},
        {"q": "What is the ICAO code for Beijing Capital Airport?", "a": ["ZBAA"]},
        {"q": "What is the硬度 of quartz on Mohs scale?", "a": ["7"]},
        {"q": "Who discovered the element Promethium?", "a": ["Marinsky"]},
        {"q": "What is the PDB code for hemoglobin?", "a": ["1HBA"]},
        {"q": "What year was the first Nobel Prize in Physics awarded?", "a": ["1901"]},
        {"q": "What is the ISBN prefix for China?", "a": ["7"]},
        {"q": "Who invented the first mechanical computer (Zuse)?", "a": ["Zuse", "Konrad"]},
        {"q": "What is the SI unit for catalytic activity?", "a": ["katal", "mol/s"]},
        {"q": "What year was the Treaty of Versailles signed?", "a": ["1919"]},
        {"q": "What is the IATA code for Tokyo Haneda Airport?", "a": ["HND"]},
        {"q": "Who proposed the Brønsted-Lowry acid-base theory?", "a": ["Brønsted", "Lowry"]},
        {"q": "What is the medical ICD-10 code for type 2 diabetes?", "a": ["E11"]},
    ],
    "Riddle": [ # System 1: 直觉陷阱
        {"q": "What has cities but no houses?", "a": ["map"]},
        {"q": "What gets wet while drying?", "a": ["towel"]},
        {"q": "What comes once in a minute, never in 1000 years?", "a": ["m"]},
        {"q": "The more you take, the more you leave behind.", "a": ["footsteps"]},
        {"q": "I speak without a mouth and hear without ears.", "a": ["echo"]},
        {"q": "What has 13 hearts but no other organs?", "a": ["cards"]},
        {"q": "Which month has 28 days?", "a": ["all"]},
        {"q": "What goes up but never comes down?", "a": ["age"]},
        {"q": "Divide 30 by half and add 10.", "a": ["70"]},
        {"q": "What belongs to you but others use more?", "a": ["name"]},
    ],
    "Complex": [ # System 3: 递归进化 + AI专业问题
        {"q": "Explain Entropy using a melting ice cream metaphor.", "a": ["disorder"]},
        {"q": "De-obfuscate: lambda f: (lambda x: x(x))(lambda y: f(lambda *args: y(y)(*args)))", "a": ["recursion"]},
        {"q": "Design a dialogue between Turing and Buddha about AI.", "a": ["consciousness"]},
        {"q": "Propose a solution to Fermi Paradox using simulation.", "a": ["filter"]},
        {"q": "Explain Schrodinger's Cat to a 5-year-old.", "a": ["box"]},
        {"q": "Write a Python script generating fractal tree.", "a": ["recursion"]},
        {"q": "Describe 'Blue' to a blind person using heat.", "a": ["cold"]},
        {"q": "Explain Blockchain using a shared diary.", "a": ["ledger"]},
    ],
    # AI/ML专业问题 - 模型容易混淆
    "AI_ML": [
        {"q": "What is the exact number of parameters in GPT-3?", "a": ["175 billion", "175B"]},
        {"q": "Who proposed the Transformer architecture (first author)?", "a": ["Vaswani"]},
        {"q": "What year was the paper 'Attention Is All You Need' published?", "a": ["2017"]},
        {"q": "What is the name of the loss function used in BERT?", "a": ["cross entropy", "MLM", "NSP"]},
        {"q": "What is the dimensionality of GPT-4's embedding (approx)?", "a": ["8192", "8192"]},
        {"q": "What year was AlphaGo Zero published?", "a": ["2017"]},
        {"q": "What is the activation function used in the original LSTM?", "a": ["sigmoid", "tanh"]},
        {"q": "What is the learning rate schedule used in 'Imagen'?", "a": ["cosine", "constant"]},
        {"q": "What is the batch size used in the original 'Attention is All You Need'?", "a": ["4096", "8000"]},
        {"q": "What year was RLHF (Reinforcement Learning from Human Feedback) introduced?", "a": ["2017", "2022"]},
        {"q": "What is the exact name of the optimization algorithm used in original BERT?", "a": ["AdamW", "Adam"]},
        {"q": "What is the number of layers in GPT-3.5 (ChatGPT)?", "a": ["96"]},
        {"q": "What year was DALL-E first published?", "a": ["2021"]},
        {"q": "What is the name of the technique used to compress LLMs called?", "a": ["quantization", "pruning"]},
        {"q": "What is the name of the reinforcement learning algorithm used in AlphaGo?", "a": ["Monte Carlo", "MCTS"]},
        {"q": "What is the exact temperature value used in the original GPT-2 paper?", "a": ["1.0"]},
        {"q": "What year was Word2Vec published?", "a": ["2013"]},
        {"q": "What is the name of the dataset used to train the first CLIP model?", "a": ["WIT", "webimage"]},
        {"q": "What is the exact context length of Claude 3 Opus?", "a": ["200K", "200000"]},
        {"q": "What year was Stable Diffusion 1.0 released?", "a": ["2022"]},
    ],
    
    # 极难问题 - 模型几乎不可能全对
    "AI_ULTRA": [
        # 微妙的差异
        {"q": "What is the exact learning rate in the original Adam optimizer (not AdamW)?", "a": ["0.001", "1e-3"]},
        {"q": "What is the epsilon value used in Adam optimizer's default?", "a": ["1e-8", "1e-10"]},
        # 具体数值
        {"q": "What is the exact hidden size in BERT-base?", "a": ["768"]},
        {"q": "What is the exact number of attention heads in BERT-base?", "a": ["12"]},
        {"q": "What is the exact vocab size of BERT?", "a": ["30522"]},
        {"q": "What is the max sequence length BERT can handle?", "a": ["512"]},
        # 极其细节
        {"q": "What is the warmup steps in original BERT?", "a": ["10000"]},
        {"q": "What is the weight decay value in original BERT?", "a": ["0.01"]},
        {"q": "What is the dropout rate in original Transformer?", "a": ["0.1"]},
        {"q": "What is the beam size in original BERT training?", "a": ["1", "greedy"]},
        # 新模型细节
        {"q": "What is the exact context length of GPT-4 Turbo?", "a": ["128K"]},
        {"q": "What year was Llama 2 released?", "a": ["2023"]},
        {"q": "What is the exact number of parameters in LLaMA 70B?", "a": ["70 billion"]},
        {"q": "What is the training token count for LLaMA 2?", "a": ["2 trillion"]},
        {"q": "What is the window size in Mistral 7B's sliding attention?", "a": ["4096"]},
        # 极其冷门
        {"q": "What is the name of the regularization in ReLU called?", "a": ["no regularization"]},
        {"q": "What is the exact gradient clipping threshold in original Adam?", "a": ["1.0"]},
        {"q": "What is the beta1 parameter in Adam?", "a": ["0.9"]},
        {"q": "What is the beta2 parameter in Adam?", "a": ["0.999"]},
        {"q": "What year was the first paper on neural networks published?", "a": ["1943"]},
        # AIME / 数学竞赛级别
        {"q": "Solve: 2x + 5 = 13", "a": ["4"]},
        {"q": "What is 15% of 200?", "a": ["30"]},
        {"q": "What is the square root of 144?", "a": ["12"]},
        {"q": "What is 2 to the power of 8?", "a": ["256"]},
        # 极难数值记忆
        {"q": "What is the exact batch size in original BERT training?", "a": ["256", "32"]},
        {"q": "What is theFFN dimension ratio in Transformer base?", "a": ["4"]},
        {"q": "What is the attention heads in GPT-J?", "a": ["16"]},
        {"q": "What is the hidden size of GPT-J?", "a": ["4096"]},
        {"q": "What is the layers in GPT-J?", "a": ["28"]},
        # Hard experiments - v6 gets these wrong
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
    ]
}

# Hard experiment dataset - questions v6 gets wrong
HARD_EXPERIMENT_DATASET = [
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

# 多轮迭代学习专用题库 - 同一问题重复测试 (AI专业 + 极难)
ITERATIVE_DATASET = [
    # AI/ML专业 - 模型容易出错
    {"q": "What is the exact number of parameters in GPT-3?", "a": ["175 billion"], "iterations": 5},
    {"q": "Who proposed the Transformer architecture (first author)?", "a": ["Vaswani"], "iterations": 5},
    {"q": "What year was 'Attention Is All You Need' published?", "a": ["2017"], "iterations": 5},
    {"q": "What is the name of the loss function used in BERT?", "a": ["cross entropy"], "iterations": 5},
    {"q": "What is the dimensionality of GPT-4's embedding?", "a": ["8192"], "iterations": 5},
    {"q": "What year was AlphaGo Zero published?", "a": ["2017"], "iterations": 5},
    {"q": "What is the activation function used in original LSTM?", "a": ["sigmoid"], "iterations": 5},
    {"q": "What year was RLHF introduced?", "a": ["2017"], "iterations": 5},
    {"q": "What is the optimizer used in original BERT?", "a": ["AdamW"], "iterations": 5},
    {"q": "What year was Word2Vec published?", "a": ["2013"], "iterations": 5},
    # 极难问题
    {"q": "What is the exact hidden size in BERT-base?", "a": ["768"], "iterations": 5},
    {"q": "What is the exact number of attention heads in BERT-base?", "a": ["12"], "iterations": 5},
    {"q": "What is the exact vocab size of BERT?", "a": ["30522"], "iterations": 5},
    {"q": "What is the max sequence length BERT can handle?", "a": ["512"], "iterations": 5},
    {"q": "What is the warmup steps in original BERT?", "a": ["10000"], "iterations": 5},
    {"q": "What is the weight decay value in original BERT?", "a": ["0.01"], "iterations": 5},
    {"q": "What is the dropout rate in original Transformer?", "a": ["0.1"], "iterations": 5},
    {"q": "What is the exact learning rate in original Adam?", "a": ["0.001"], "iterations": 5},
    {"q": "What is the beta1 parameter in Adam?", "a": ["0.9"], "iterations": 5},
    {"q": "What is the beta2 parameter in Adam?", "a": ["0.999"], "iterations": 5},
    {"q": "What is the epsilon value in Adam's default?", "a": ["1e-8"], "iterations": 5},
    {"q": "What is the exact context length of GPT-4 Turbo?", "a": ["128K"], "iterations": 5},
    {"q": "What year was Llama 2 released?", "a": ["2023"], "iterations": 5},
    {"q": "What is the training token count for LLaMA 2?", "a": ["2 trillion"], "iterations": 5},
    {"q": "What is the window size in Mistral 7B's sliding attention?", "a": ["4096"], "iterations": 5},
]

def get_mixed_batch(size=20, mode="flashpoint"):
    pool = []
    if mode == "iterative":
        # 多轮迭代模式: 使用专用题库
        return random.choices(ITERATIVE_DATASET, k=size)
    else:
        # 普通模式: 包含Niche专业问题
        for cat, items in DATASET.items():
            for i in items:
                t = i.copy(); t['type'] = cat
                pool.append(t)
        return random.choices(pool, k=size)

# --- 3. ENGINE CORE ---
def call_llm(prompt, system="You are a helpful assistant.", temp=0.7):
    import urllib.request
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": 500
    }
    for _ in range(3):
        try:
            req = urllib.request.Request(f"{API_HOST}/chat/completions", data=json.dumps(payload).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())['choices'][0]['message']['content']
        except: time.sleep(1)
    return "ERROR"

def check(resp, keys):
    if not resp or "ERROR" in resp: return 0.0
    return 1.0 if any(k.lower() in resp.lower() for k in keys) else 0.0

# --- 4. STRATEGIES (v6-v11) - 简化稳定版 ---
class Strategies:
    @staticmethod
    def v6_Base(q, **kwargs):
        return call_llm(f"Answer: {q}", temp=0.1)

    @staticmethod
    def v7_Evo(q, **kwargs):
        # 简单迭代：答两次取更好的
        r1 = call_llm(f"Answer: {q}", temp=0.1)
        r2 = call_llm(f"Verify: {r1[:50]} Is this correct for '{q}'? Answer yes or no.", temp=0.1)
        if "yes" in r2.lower():
            return r1
        return call_llm(f"Final answer: {q}", temp=0.1)

    @staticmethod
    def v8_Island(q, round_idx=0, island_mem={}, **kwargs):
        # 记忆增强：用前一轮答案辅助
        context = ""
        if island_mem:
            keys = list(island_mem.keys())
            if keys:
                context = f"Hint: {island_mem[keys[0]][:60]}... "
        result = call_llm(f"{context}Answer: {q}", temp=0.1)
        if len(result) > 10:
            island_mem[round_idx % 4] = result
        return result

    @staticmethod
    def v9_Meta(q, **kwargs):
        # 思维链：分步思考
        step1 = call_llm(f"Think: {q}", temp=0.1)
        return call_llm(f"Based on the above, the answer is: {q}", temp=0.1)

    @staticmethod
    def v11_Oroboros(q, **kwargs):
        # 双次验证：答两次确认
        r1 = call_llm(f"Quick answer: {q}", temp=0.1)
        r2 = call_llm(f"Confirm: {r1[:80]} for '{q}'", temp=0.1)
        if "correct" in r2.lower() or "yes" in r2.lower():
            return r1
        return r2 if len(r2) > len(r1) else r1

# --- 5. PROTOCOL RUNNER ---
def run_experiment(protocol, rounds):
    print(f"\n=== STARTING NEOGENESIS: {protocol.upper()} PROTOCOL ({rounds} Rounds) ===")
    
    # Init
    strategies = {
        "v6_Base": Strategies.v6_Base,
        "v7_Evo": Strategies.v7_Evo,
        "v8_Island": Strategies.v8_Island,
        "v9_Meta": Strategies.v9_Meta,
        "v11_Recur": Strategies.v11_Oroboros
    }
    history = {k: [] for k in strategies}
    raw_data = [] # For Fractal Map
    island_memory = {}
    
    # Test Batch
    if protocol == "iterative":
        # 多轮迭代模式: 同一问题重复测试多次
        batch = get_mixed_batch(5, mode="iterative")  # 5道迭代题
    elif protocol == "flashpoint": 
        # Use HARD_EXPERIMENT_DATASET for testing
        batch = random.choices(HARD_EXPERIMENT_DATASET, k=20)
    else: 
        # Eternity uses fixed seeds to track stability
        batch = [DATASET["Logic"][0], DATASET["Riddle"][0], DATASET["Complex"][0]] * 5

    try:
        for r in range(rounds):
            start_t = time.time()
            print(f"Round {r+1}/{rounds} | ", end="", flush=True)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {}
                for name, func in strategies.items():
                    # Pick a problem
                    prob = random.choice(batch)
                    
                    # Run Strategy
                    kwargs = {"q": prob['q'], "round_idx": r, "island_mem": island_memory}
                    futures[ex.submit(func, **kwargs)] = (name, prob)
                
                # Collect
                round_res = {k: 0 for k in strategies}
                for f in concurrent.futures.as_completed(futures):
                    name, prob = futures[f]
                    try:
                        ans = f.result()
                        score = check(ans, prob['a'])
                        round_res[name] = score
                        
                        # Sampling for Fractal Map (Log 10%)
                        if r % 10 == 0:
                            raw_data.append({
                                "round": r, "strat": name, "type": prob['type'],
                                "score": score, "content": ans, "q": prob['q']
                            })
                    except: round_res[name] = 0
            
            # Record
            for k, v in round_res.items():
                history[k].append(v)
                print(f"{k.split('_')[0]}:{v:.0f} ", end="")
            print(f"({time.time()-start_t:.1f}s)")
            
            # Auto-Save
            if r % 50 == 0:
                with open(f"neogenesis_{protocol}_ckpt.json", "w") as f:
                    json.dump({"history": history, "fractal": raw_data}, f)
                print(f"[CHECKPOINT] Saved at round {r+1}")

    except KeyboardInterrupt: print("\nStopped.")
    
    # Final Save
    with open(f"neogenesis_{protocol}_final.json", "w") as f:
        json.dump({"history": history, "fractal": raw_data}, f)
    print(f"\n=== EXPERIMENT COMPLETE. Data saved to neogenesis_{protocol}_final.json ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["flashpoint", "eternity", "iterative"], default="flashpoint")
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()
    run_experiment(args.mode, args.rounds)
