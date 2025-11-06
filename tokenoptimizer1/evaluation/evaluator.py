#
# LLM Evaluation Script for LM Studio
#
# This script evaluates models hosted on an LM Studio local server, measuring
# performance, memory usage, and cost-efficiency metrics.
#
# The cost calculation functions are based on the paper "Efficient Agents:
# Building Effective Agents While Reducing Cost"[cite: 3].
#
# The code for API interaction and GPU memory measurement is a standard
# implementation and is NOT from the provided paper.
#
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.llm_interface import query_llm, build_prompt
from gating.token_gater import gate, apply_stepwise_gating, apply_hybrid_scoring_gating, apply_diversity_gating, apply_token_budget_gating

import time
import math
from openai import OpenAI
import pandas as pd

# Attempt to import pynvml for GPU memory measurement
try:
    import pynvml
except ImportError:
    print("Warning: pynvml library not found. GPU memory measurement will be disabled.")
    print("To enable memory measurement, run: pip install pynvml")
    pynvml = None

# ==============================================================================
# 1. API and System Measurement Functions
# ==============================================================================

def query_token_optimiser(prompt_text, gating_mode=None):
    """Run the full Token Optimiser (gating + LLM) for evaluation."""
    start_time = time.perf_counter()

    gated_prompt = prompt_text
    if gating_mode == "stepwise":
        gated_prompt = apply_stepwise_gating(prompt_text)
    elif gating_mode == "hybrid":
        gated_prompt = apply_hybrid_scoring_gating(prompt_text)
    elif gating_mode == "diversity":
        gated_prompt = apply_diversity_gating(prompt_text)
    elif gating_mode == "token_budget":
        gated_prompt = apply_token_budget_gating(prompt_text)
    elif gating_mode == "original_gating":
        context = gate(prompt_text)
        gated_prompt = build_prompt(context, prompt_text)

    if gating_mode not in ["original_gating"]:
        response_text = query_llm(gated_prompt)
    else:
        response_text = query_llm(gated_prompt)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    # Token counting
    from utils.text_utils import count_tokens
    prompt_tokens = count_tokens(gated_prompt)
    completion_tokens = count_tokens(response_text)
    tps = completion_tokens / total_time if total_time > 0 else 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_sec": total_time,
        "tokens_per_second": tps,
        "response_text": response_text
    }

def get_gpu_memory_utilization(device_id: int = 0):
    """Checks the GPU memory utilization for a specific NVIDIA GPU."""
    if pynvml is None: return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"used_mib": mem_info.used / (1024 ** 2)}
    except pynvml.NVMLError:
        return None

# ==============================================================================
# 2. Cost Calculation Functions (from the provided paper)
# ==============================================================================

def calculate_inference_cost(num_input_tokens: int, cost_per_input_token: float,
                             num_output_tokens: int, cost_per_output_token: float) -> float:
    """
    Calculates the cost of an inference attempt, C_m(p).
    This function implements the formula:
    C_m(p) = n_in(m,p) * c_in(m) + n_out(m,p) * c_out(m).
    """
    return (num_input_tokens * cost_per_input_token) + (num_output_tokens * cost_per_output_token)

def calculate_cost_of_pass(total_inference_cost: float, success_rate: float) -> float:
    """
    Calculates the cost-of-pass metric, v(m,p).
    This function implements the formula v(m,p) = C_m(p) / R_m(p).
    """
    if success_rate == 0:
        # As noted in the paper, cost-of-pass is infinity if accuracy is zero[cite: 114].
        return math.inf
    return total_inference_cost / success_rate

# ==============================================================================
# 3. Main Evaluation Orchestrator
# ==============================================================================

def evaluate_model(model_config, client, query):
    """Runs the full evaluation suite for a single model."""
    print("\n" + "="*80)
    print(f"Starting Evaluation for Model: {model_config['name']}")
    print("="*80)
    
    # --- Performance Evaluation ---
    sample_prompts = [query]
    
    gating_modes = [None, "original_gating", "stepwise", "hybrid", "diversity", "token_budget"]
    all_results = []

    for mode in gating_modes:
        results = []
        print(f"\nRunning {len(sample_prompts)} sample prompts with gating mode: {mode or 'baseline'}...")
        for i, prompt in enumerate(sample_prompts):
            print(f"  Querying prompt {i+1}/{len(sample_prompts)}...")
            result = query_token_optimiser(prompt, gating_mode=mode)
            if result:
                result['gating_mode'] = mode or 'baseline'
                results.append(result)
                print(f"    -> Response generated in {result['total_time_sec']:.2f}s ({result['tokens_per_second']:.2f} TPS)")
            else:
                print("    -> Failed to get response. Aborting evaluation for this model.")
                return None
        all_results.extend(results)
    
    # --- Aggregate Performance Metrics ---
    df = pd.DataFrame(all_results)
    
    # --- Get Manual Success Rate (R_m(p)) ---
    success_rate = 1.0 # Hardcoded for now

    df['success_rate'] = success_rate

    # --- Cost Calculation ---
    df['hypothetical_cost'] = df.apply(
        lambda row: calculate_inference_cost(
            row['prompt_tokens'],
            model_config['hypothetical_cost_per_input_token'],
            row['completion_tokens'],
            model_config['hypothetical_cost_per_output_token']
        ), axis=1
    )
    df['cost_of_pass'] = df.apply(
        lambda row: calculate_cost_of_pass(row['hypothetical_cost'], row['success_rate']), axis=1
    )

    # --- Store all results ---
    final_report = {
        "Model Name": model_config['name'],
    }
    
    print("\n--- Evaluation Summary ---")
    for key, value in final_report.items():
        print(f"{key:<25}: {value}")

    print("\n--- Detailed Gating Results ---")
    print(df[['gating_mode', 'prompt_tokens', 'completion_tokens', 'total_time_sec', 'tokens_per_second', 'hypothetical_cost', 'cost_of_pass']].round(4).to_string(index=False))
    
    return df

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

def run_evaluation(query):
    # --- Model Configurations ---
    MODEL_1_CONFIG = {
        "name": "liquid/lfm2-1.2b",
        "hypothetical_cost_per_input_token": 0.20 / 1_000_000,
        "hypothetical_cost_per_output_token": 1.00 / 1_000_000
    }

    models_to_evaluate = [MODEL_1_CONFIG]
    all_results_df = pd.DataFrame()
    
    # Initialize LM Studio Client
    lm_studio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    
    # Initialize NVML for memory reading
    if pynvml:
        pynvml.nvmlInit()
    
    try:
        for config in models_to_evaluate:
            # User interaction to load the model
            input(f"Please load the '{config['name']}' model in LM Studio and start the server.\n"
                "Press Enter when you are ready to begin the evaluation...")

            # --- Memory Measurement (Before) ---
            mem_before = get_gpu_memory_utilization()
            if mem_before:
                print(f"Initial VRAM Usage: {mem_before['used_mib']:.2f} MiB")

            report_df = evaluate_model(config, lm_studio_client, query)
            if report_df is not None:
                report_df['model_name'] = config['name']
                all_results_df = pd.concat([all_results_df, report_df])

            # --- Memory Measurement (After) ---
            mem_after = get_gpu_memory_utilization()
            model_vram_footprint = (mem_after['used_mib'] - mem_before['used_mib']) if mem_before and mem_after else 0
            print(f"VRAM Footprint (MiB): {model_vram_footprint:.2f}")

    finally:
        # Ensure NVML is shut down properly
        if pynvml:
            pynvml.nvmlShutdown()

    # --- Final Comparison Table ---
    if not all_results_df.empty:
        print("\n" + "="*80)
        print("Final Comparison Report")
        print("="*80)
        print(all_results_df[['model_name', 'gating_mode', 'prompt_tokens', 'completion_tokens', 'total_time_sec', 'tokens_per_second', 'hypothetical_cost', 'cost_of_pass']].round(4).to_string(index=False))

if __name__ == "__main__":
    default_query = "What are the top 3 benefits of adopting a microservices architecture?"
    run_evaluation(default_query)