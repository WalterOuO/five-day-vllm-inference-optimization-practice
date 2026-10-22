import argparse
import json
import os
import time
import yaml
import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams

def compute_model_weight_vram_gb(model_id):
    """透過 AutoConfig 動態計算模型權重理論顯存佔用 (GB)"""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    vocab_size = getattr(config, "vocab_size", 151936)
    hidden_size = getattr(config, "hidden_size", 2048)
    intermediate_size = getattr(config, "intermediate_size", 11008)
    num_layers = getattr(config, "num_hidden_layers", 36)
    num_heads = getattr(config, "num_attention_heads", 16)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    
    head_dim = hidden_size // num_heads
    q_proj = hidden_size * (num_heads * head_dim)
    k_proj = hidden_size * (num_kv_heads * head_dim)
    v_proj = hidden_size * (num_kv_heads * head_dim)
    out_proj = (num_heads * head_dim) * hidden_size
    attn_params = q_proj + k_proj + v_proj + out_proj
    mlp_params = 3 * hidden_size * intermediate_size
    layer_params = attn_params + mlp_params
    total_params = (vocab_size * hidden_size) + (num_layers * layer_params) + (vocab_size * hidden_size)
    
    # FP16 = 2 Bytes per Parameter
    return round((total_params * 2.0) / (1024 ** 3), 2)

def build_stitched_prompts(json_path, concurrency, stitching_count):
    """
    從 benchmark_prompts.json 讀取 32 個獨立 Prompt，
    透過滾動滑窗拼接出 concurrency 個、長度約為 16K Tokens 的獨立長文本。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        base_prompts = json.load(f)
    
    num_base = len(base_prompts)
    stitched_prompts = []
    
    for i in range(concurrency):
        # 滾動挑選 stitching_count (16) 個 prompt 進行拼接
        selected = [base_prompts[(i + j) % num_base] for j in range(stitching_count)]
        stitched_text = "\\n\\n".join(selected)
        stitched_prompts.append(stitched_text)
        
    return stitched_prompts

def run_experiment(mode, config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_id = cfg["model_id"]
    gpu_util = cfg["gpu_memory_utilization"]
    max_model_len = cfg["max_model_len"]
    concurrency = cfg["concurrency"]
    stitching_count = cfg["prompt_stitching_count"]
    max_tokens = cfg["max_tokens"]
    
    cpu_offload_gb = cfg["baseline_offload_gb"] if mode == "baseline" else cfg["optimization_offload_gb"]
    
    print(f"\\n==================================================")
    print(f"🚀 [Day 4] 執行 Mode: {mode.upper()} | CPU Offload GB: {cpu_offload_gb} GB")
    print(f"==================================================")
    
    # 讀取並拼接 16K 長文本 Prompts
    prompt_json = "data/prompts/benchmark_prompts.json"
    prompts = build_stitched_prompts(prompt_json, concurrency, stitching_count)
    
    # 計算顯存指標
    model_vram = compute_model_weight_vram_gb(model_id)
    _, total_vram_bytes = torch.cuda.mem_get_info()
    total_vram_gb = total_vram_bytes / (1024 ** 3)
    gpu_reserved_vram = round(total_vram_gb * gpu_util, 2)
    gpu_kv_cache_vram = round(max(0, gpu_reserved_vram - model_vram - 0.5), 2)
    
    # 預估 KV Cache 總需求 (16 併發 x 16K len x 0.035MB/token)
    est_kv_demand_gb = round((concurrency * 16000 * 0.03515) / 1024, 2)
    est_cpu_swapped_gb = round(max(0, est_kv_demand_gb - gpu_kv_cache_vram), 2)
    
    print(f"📊 [Memory Breakdown Preview]")
    print(f"   - GPU 實體總顯存: {total_vram_gb:.2f} GB")
    print(f"   - GPU 預留顯存上限 (util={gpu_util}): {gpu_reserved_vram:.2f} GB")
    print(f"   - 模型權重佔用 (FP16): {model_vram:.2f} GB")
    print(f"   - GPU KV Cache 容量: ~{gpu_kv_cache_vram:.2f} GB")
    print(f"   - 16K x 16 併發 KV Cache 總需求: ~{est_kv_demand_gb:.2f} GB")
    print(f"   - 預計溢出至 CPU Offloading/Swapping: ~{est_cpu_swapped_gb:.2f} GB")
    
    success = False
    error_msg = ""
    elapsed_time = 0.0
    throughput = 0.0
    
    try:
        # 相容新版 vLLM 的 cpu_offload_gb 參數
        llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_util,
            cpu_offload_gb=cpu_offload_gb,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True
        )
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        elapsed_time = time.time() - start_time
        
        total_generated_tokens = sum([len(out.outputs[0].token_ids) for out in outputs])
        throughput = total_generated_tokens / elapsed_time
        success = True
        print(f"\\n✅ [成功完成] 耗時: {elapsed_time:.2f}s | 吞吐量: {throughput:.2f} tokens/s")
        
    except Exception as e:
        error_msg = str(e)
        print(f"\\n❌ [推論失敗/OOM/崩潰] 原因: {error_msg}")
        
    result_data = {
        "mode": mode,
        "cpu_offload_gb": cpu_offload_gb,
        "success": success,
        "error_msg": error_msg,
        "elapsed_time_s": round(elapsed_time, 2),
        "throughput_tps": round(throughput, 2),
        "model_vram_gb": model_vram,
        "gpu_kv_cache_vram_gb": gpu_kv_cache_vram,
        "est_kv_demand_gb": est_kv_demand_gb,
        "est_cpu_swapped_gb": est_cpu_swapped_gb if success else 0.0
    }
    
    out_file = f"results/raw_logs/day4_{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
        
    print(f"💾 實驗結果已儲存至: {out_file}\\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "offload"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_experiment(args.mode, args.config)