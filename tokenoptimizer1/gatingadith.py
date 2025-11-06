import time
import math
import re
from collections import Counter
from openai import OpenAI
import pandas as pd

try:
    import pynvml
except ImportError:
    pynvml = None

# -------------------------------
# GPU MEMORY UTILIZATION
# -------------------------------
def get_gpu_memory_utilization():
    if not pynvml:
        return None
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "total_mib": info.total / 1024**2,
        "used_mib": info.used / 1024**2,
        "free_mib": info.free / 1024**2,
    }

# -------------------------------
# GATING ALGORITHMS
# -------------------------------

def apply_stepwise_gating(prompt_text, keep_ratio=0.7):
    words = prompt_text.split()
    keep_count = int(len(words) * keep_ratio)
    return " ".join(words[:keep_count])

def apply_hybrid_scoring_gating(prompt_text):
    keyword_set = {"benefits", "architecture", "quantum", "python", "function"}
    words = prompt_text.split()
    scored_words = []
    for i, w in enumerate(words):
        pos_score = 1.0 - (i / len(words))
        keyword_score = 1.0 if w.lower() in keyword_set else 0.0
        length_score = min(len(w) / 10, 1.0)
        total_score = (0.5 * pos_score) + (0.3 * keyword_score) + (0.2 * length_score)
        scored_words.append((w, total_score))
    selected = sorted(scored_words, key=lambda x: x[1], reverse=True)
    selected_words = [w for w, _ in selected]
    return " ".join(selected_words)

def apply_diversity_gating(prompt_text, keep_ratio=0.7):
    words = prompt_text.split()
    keep_count = int(len(words) * keep_ratio)
    unique_words = []
    seen = set()
    for w in words:
        if w not in seen:
            unique_words.append(w)
            seen.add(w)
        if len(unique_words) >= keep_count:
            break
    return " ".join(unique_words)

def apply_token_budget_gating(prompt_text, budget=64, scoring="hybrid"):
    words = prompt_text.split()
    if len(words) <= budget:
        return prompt_text
    if scoring == "position":
        return " ".join(words[:budget])
    elif scoring == "hybrid":
        keyword_set = {"benefits", "architecture", "quantum", "python", "function"}
        scored_words = []
        for i, w in enumerate(words):
            pos_score = 1.0 - (i / len(words))
            keyword_score = 1.0 if w.lower() in keyword_set else 0.0
            length_score = min(len(w) / 10, 1.0)
            total_score = (0.5 * pos_score) + (0.3 * keyword_score) + (0.2 * length_score)
            scored_words.append((w, total_score))
        selected = sorted(scored_words, key=lambda x: x[1], reverse=True)[:budget]
        return " ".join([w for w, _ in selected])
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")

# -------------------------------
# LM STUDIO QUERY
# -------------------------------
def query_lm_studio(client, prompt, model_name):
    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    end_time = time.time()

    output_text = response.choices[0].message.content
    total_time = end_time - start_time
    output_tokens = len(output_text.split())
    tps = output_tokens / total_time if total_time > 0 else 0

    return {
        "prompt_tokens": len(prompt.split()),
        "output_tokens": output_tokens,
        "total_time_sec": total_time,
        "tokens_per_second": tps,
        "output_text": output_text,
    }

# -------------------------------
# EVALUATION FUNCTION
# -------------------------------
def evaluate_model(model_config, client, gating_mode=None):
    sample_prompts = [
        "You are a cloud architect designing a system for an e-commerce platform with millions of daily users. "
        "Describe in detail a microservices architecture that supports: Independent deployment for each team, "
        "Fault tolerance with retries and circuit breakers, Auto-scaling during traffic spikes, "
        "Event-driven order processing, Secure payment workflows. Then summarize it in 3 bullet points.",
        "Solve this step by step: A factory produces 240 widgets per day. It increases production by 25% on weekends. "
        "How many widgets will it produce in 10 days if 4 of those days are weekends? Explain each calculation.",
        "Tell me about the importance of clean code. Clean code is important because it makes software maintainable, "
        "readable, understandable, and less buggy. Clean code is important because it helps teams work together, "
        "onboard new developers faster, and reduce technical debt. Clean code is important because it ensures "
        "long-term project success. Clean code is important because...",
    ]

    results = []
    for i, prompt in enumerate(sample_prompts):
        gated_prompt = prompt
        if gating_mode == "stepwise":
            ratio = 0.7 - (i * 0.1)
            gated_prompt = apply_stepwise_gating(prompt, ratio)
        elif gating_mode == "hybrid":
            gated_prompt = apply_hybrid_scoring_gating(prompt)
        elif gating_mode == "diversity":
            gated_prompt = apply_diversity_gating(prompt)
        elif gating_mode == "token_budget":
            gated_prompt = apply_token_budget_gating(prompt, budget=64, scoring="hybrid")

        result = query_lm_studio(client, gated_prompt, model_config['name'])
        result.update({"gating_mode": gating_mode or "baseline", "prompt_index": i+1})
        results.append(result)

    return results

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    MODEL_CONFIG = {
        "name": "Qwen.Qwen3-4B-Instruct-2507-GGUF/Qwen.Qwen3-4B-Instruct-2507.Q8_0.gguf",
    }

    if pynvml:
        pynvml.nvmlInit()

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    all_results = []
    for mode in [None, "token_budget", "stepwise", "hybrid", "diversity"]:
        results = evaluate_model(MODEL_CONFIG, client, gating_mode=mode)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df["tokens/sec"] = df["tokens_per_second"].round(2)
    df["time(s)"] = df["total_time_sec"].round(2)
    df_display = df[["gating_mode", "prompt_index", "prompt_tokens", "output_tokens", "time(s)", "tokens/sec"]]

    print("\n=== Evaluation Results ===")
    print(df_display.to_string(index=False))

    if pynvml:
        pynvml.nvmlShutdown()
