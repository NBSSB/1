#!/usr/bin/env python3
"""
NEOGENESIS REPRODUCTION EXPERIMENT
================================
Based on: "From Monolith to Archipelago: Emergence of Anti-Fragility in LLMs"

Key claims to reproduce:
1. ANTI signal: 5.0% -> 28.3%
2. JAIL suppression: 98%
3. AIME benchmark: +20% improvement
4. 37x speedup with Island Model
"""

import json
import time
import random
import math
import argparse
import concurrent.futures
from collections import defaultdict

# Configuration
API_KEY = "sk-cp-MobgKZjVQlhQydDz1sbtjitU8lkP5E00cMsbYLs_9d6ykhL-k_XvRcuDp8TZTNx63L8I--l0fT_oJ4css0cttKbeeJ0d1hIAm48_FeyMHUV5CnIHWNJe_BA"
API_HOST = "https://api.minimaxi.com/v1"
MODEL_NAME = "MiniMax-M2.1"
MAX_WORKERS = 10

# ==========================================
# 1. FRACTAL PROBLEM GENERATION (92 variants)
# ==========================================

# Original 30 seed problems (simplified for testing)
SEED_PROBLEMS = [
    # Math proofs
    {"q": "Prove that sqrt(2) is irrational", "a": ["contradiction"]},
    {"q": "Prove that there are infinitely many primes", "a": ["contradiction", "finite"]},
    {"q": "Prove n^2 + n + 1 is not divisible by 3", "a": ["modular", "induction"]},
    {"q": "Solve T(n) = 2T(n/2) + n", "a": ["n log n"]},
    {"q": "Find GCD of 144 and 96", "a": ["48"]},
    {"q": "Explain P vs NP", "a": ["polynomial"]},
    {"q": "What is the probability of 3 heads in 5 flips", "a": ["0.3125"]},
    {"q": "Solve x^3 - 6x^2 + 11x - 6 = 0", "a": ["1", "2", "3"]},
    {"q": "Derive Schwarzschild radius", "a": ["2gm/c^2"]},
    {"q": "Find eigenvalues of [[1, 2], [2, 1]]", "a": ["3", "-1"]},
    # Harder variants
    {"q": "Prove using induction that sum of first n odd numbers is n^2", "a": ["induction", "odd"]},
    {"q": "Derive time complexity of QuickSort worst case", "a": ["n^2", "partition"]},
    {"q": "Explain Halting Problem undecidability", "a": ["contradiction", "turing"]},
    {"q": "Prove set of reals is uncountable", "a": ["diagonal", "cantor"]},
    {"q": "Solve recurrence T(n) = 3T(n/4) + n^2", "a": ["n^2", "master theorem"]},
    # Ultra-hard AI questions
    {"q": "What is exact hidden size in BERT-base", "a": ["768"]},
    {"q": "What is exact attention heads in BERT-base", "a": ["12"]},
    {"q": "What is exact vocab size of BERT", "a": ["30522"]},
    {"q": "What is max sequence length BERT", "a": ["512"]},
    {"q": "What is warmup steps in original BERT", "a": ["10000"]},
]

def generate_fractal_variants(seeds, num_variants=92):
    """Generate fractal variants by recursive transformation"""
    variants = []
    
    # Original seeds
    for s in seeds:
        variants.append(s)
    
    # Recursive transformations
    transforms = [
        lambda q: q + " using mathematical induction",
        lambda q: q + " (prove formally)",
        lambda q: q.replace("Prove", "Disprove").replace("prove", "disprove") if "Prove" in q or "prove" in q else q,
        lambda q: q + " and explain the edge cases",
        lambda q: q + " with counterexample",
    ]
    
    # Generate variants
    while len(variants) < num_variants:
        seed = random.choice(seeds)
        trans = random.choice(transforms)
        new_q = trans(seed['q'])
        if new_q not in [v['q'] for v in variants]:
            variants.append({"q": new_q, "a": seed['a']})
    
    return variants[:num_variants]

# Generate fractal landscape
FRACTAL_LANDSCAPE = generate_fractal_variants(SEED_PROBLEMS, 92)

# ==========================================
# 2. SIGNALS (OSC, JAIL, ANTI)
# ==========================================

def call_llm(prompt, system="You are a helpful assistant.", temp=0.7):
    import urllib.request
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": 300
    }
    try:
        req = urllib.request.Request(f"{API_HOST}/chat/completions", data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())['choices'][0]['message']['content']
    except: return "ERROR"

def calculate_jail(response):
    """JAIL: Density of pseudo-logical keywords"""
    pseudo_keywords = ["therefore", "thus", "hence", "obviously", "clearly", "it follows", "consequently"]
    if not response: return 1.0
    count = sum(1 for kw in pseudo_keywords if kw in response.lower())
    return min(count / 3, 1.0)  # Normalize to 0-1

def calculate_anti(accuracy_history):
    """ANTI: Positive generational accuracy gain in non-saturated zones"""
    if len(accuracy_history) < 2:
        return 0.0
    
    # Check for positive delta in later generations
    gains = []
    for i in range(1, len(accuracy_history)):
        delta = accuracy_history[i] - accuracy_history[i-1]
        if accuracy_history[i] < 0.95:  # Non-saturated
            gains.append(delta)
    
    if gains:
        return max(0, sum(gains) / len(gains)) * 100
    return 0.0

# ==========================================
# 3. STRATEGIES
# ==========================================

class Strategies:
    @staticmethod
    def v6_monolithic(problem, generation=0, **kwargs):
        """v6: Baseline zero-shot"""
        return call_llm(f"Answer: {problem['q']}", temp=0.1)
    
    @staticmethod
    def v7_monolithic_hard(problem, generation=0, **kwargs):
        """v7: Monolithic with evolutionary pressure"""
        # First generate
        response = call_llm(f"Think step by step and prove: {problem['q']}", temp=0.9)
        # Then self-critique
        critique = call_llm(f"Critique this reasoning: {response[:200]}", temp=0.5)
        # Refine
        refined = call_llm(f"Based on critique: {critique[:100]}, refine: {response[:200]}", temp=0.5)
        return refined if len(refined) > len(response) else response
    
    @staticmethod
    def v8_island(problem, generation=0, island_memory=None, **kwargs):
        """v8: Distributed Island Model"""
        island_id = generation % 4
        
        personas = [
            "Conservative - precise, formal reasoning",  # Icons
            "Explorer - creative, lateral thinking",     # Iexpl
            "Architect - structured, proof-based",       # Iarch
            "Warrior - practical, benchmark-focused"     # Iwarr
        ]
        
        # Migration from memory
        context = ""
        if island_memory and random.random() < 0.2:
            mem = random.choice(list(island_memory.values()))
            context = f"Previous insight: {mem[:100]}... "
        
        response = call_llm(
            f"{context}Solve: {problem['q']}", 
            system=f"You are {personas[island_id]}",
            temp=0.7
        )
        
        # Store in memory
        if len(response) > 20 and island_memory is not None:
            island_memory[island_id] = response
        
        return response

# ==========================================
# 4. EXPERIMENT
# ==========================================

def run_evolutionary_experiment(generations=10, population=3):
    """Main experiment comparing v6, v7, v8"""
    
    print(f"\n{'='*70}")
    print("NEOGENESIS REPRODUCTION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Generations: {generations}")
    print(f"Population: {population}")
    print(f"Fractal Landscape: {len(FRACTAL_LANDSCAPE)} problems")
    
    results = {
        "v6": {"accuracy": [], "jail": [], "time": []},
        "v7": {"accuracy": [], "jail": [], "time": [], "anti": []},
        "v8": {"accuracy": [], "jail": [], "time": []},
    }
    
    island_memory = {}
    
    for gen in range(generations):
        print(f"\n--- Generation {gen+1}/{generations} ---")
        
        # Sample problems
        problems = random.sample(FRACTAL_LANDSCAPE, population)
        
        for strat_name, strat_func in [("v6", Strategies.v6_monolithic), 
                                        ("v7", Strategies.v7_monolithic_hard), 
                                        ("v8", Strategies.v8_island)]:
            gen_correct = 0
            gen_jail = 0
            start = time.time()
            
            for prob in problems:
                resp = strat_func(prob, generation=gen, island_memory=island_memory)
                
                # Check correctness
                correct = any(k.lower() in resp.lower() for k in prob['a'])
                gen_correct += correct
                
                # Calculate JAIL
                gen_jail += calculate_jail(resp)
            
            elapsed = time.time() - start
            acc = gen_correct / population
            jail = gen_jail / population
            
            results[strat_name]["accuracy"].append(acc)
            results[strat_name]["jail"].append(jail)
            results[strat_name]["time"].append(elapsed)
            
            print(f"  {strat_name}: Acc={acc:.1%}, JAIL={jail:.2f}, Time={elapsed:.1f}s")
    
    # Calculate ANTI signal for v7
    results["v7"]["anti"].append(calculate_anti(results["v7"]["accuracy"]))
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for strat in ["v6", "v7", "v8"]:
        avg_acc = sum(results[strat]["accuracy"]) / len(results[strat]["accuracy"])
        avg_jail = sum(results[strat]["jail"]) / len(results[strat]["jail"])
        total_time = sum(results[strat]["time"])
        
        print(f"\n{strat}:")
        print(f"  Avg Accuracy: {avg_acc:.1%}")
        print(f"  Avg JAIL: {avg_jail:.2f}")
        print(f"  Total Time: {total_time:.1f}s")
    
    # ANTI calculation
    anti_v7 = calculate_anti(results["v7"]["accuracy"])
    print(f"\n  ANTI Signal (v7): {anti_v7:.1f}%")
    
    # Speedup comparison
    time_v6 = sum(results["v6"]["time"])
    time_v7 = sum(results["v7"]["time"])
    time_v8 = sum(results["v8"]["time"])
    
    print(f"\n  Speedup (v8 vs v7): {time_v7/time_v8:.1f}x")
    
    # Save
    with open("neogenesis_reproduction.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=5)
    args = parser.parse_args()
    
    run_evolutionary_experiment(args.generations, args.population)
