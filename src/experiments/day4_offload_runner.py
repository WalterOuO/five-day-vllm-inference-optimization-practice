import os

# 強制切回 V0 引擎
os.environ["VLLM_USE_V1"] = "0"

import argparse
import gc
import json
import time
import yaml
import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams

def cleanup_cuda():
    """清理 CUDA 顯存與垃圾回收"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_real_allocated_vram_gb():
    """使用 CUDA Driver API 抓取整體 GPU 當前真實已分配顯存 (GB)"""
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated_bytes = total_bytes - free_bytes
        return round(allocated_bytes / (1024 ** 3), 2)
    return 0.0

def compute_model_weight_vram_gb(model_id):
    """透過 AutoConfig 計算模型權重理論顯存 (GB)"""
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
    
    return round((total_params * 2.0) / (1024 ** 3), 2)

def build_stitched_prompts(json_path, concurrency, stitching_count):
    """拼接 12K 長文本 Prompts"""
    with open(json_path, "r", encoding="utf-8") as f:
        base_prompts = json.load(f)
    
    num_base = len(base_prompts)
    stitched_prompts = []
    for i in range(concurrency):
        selected = [base_prompts[(i + j) % num_base] for j in range(stitching_count)]
        stitched_prompts.append("\\n\\n".join(selected))
    return stitched_prompts

def run_experiment(mode, config_path):
    cleanup_cuda()
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_id = cfg["model_id"]
    gpu_util = cfg["gpu_memory_utilization"]
    max_model_len = cfg["max_model_len"]
    concurrency = cfg["concurrency"]
    stitching_count = cfg["prompt_stitching_count"]
    max_tokens = cfg["max_tokens"]
    
    use_chunked = cfg["baseline_chunked"] if mode == "baseline" else cfg["optimization_chunked"]
    max_batched_tokens = cfg["max_num_batched_tokens"]
    
    print(f"\\n==================================================")
    print(f"🚀 [Day 4] Mode: {mode.upper()} | Chunked KV Allocation: {use_chunked}")
    print(f"==================================================")
    
    prompts = build_stitched_prompts("data/prompts/benchmark_prompts.json", concurrency, stitching_count)
    
    model_vram = compute_model_weight_vram_gb(model_id)
    _, total_vram_bytes = torch.cuda.mem_get_info()
    total_vram_gb = total_vram_bytes / (1024 ** 3)
    gpu_reserved_vram = round(total_vram_gb * gpu_util, 2)
    
    success = False
    error_msg = ""
    elapsed_time = 0.0
    throughput = 0.0
    measured_physical_vram_gb = 0.0
    
    try:
        llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_util,
            cpu_offload_gb=0,
            enable_chunked_prefill=use_chunked,
            max_num_batched_tokens=max_batched_tokens if use_chunked else None,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True
        )
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        
        elapsed_time = time.time() - start_time
        
        # 透過 CUDA Driver API 抓取 vLLM 在推論時整體 GPU 真實實體顯存佔用
        measured_physical_vram_gb = get_real_allocated_vram_gb()
        
        total_generated_tokens = sum([len(out.outputs[0].token_ids) for out in outputs])
        throughput = total_generated_tokens / elapsed_time
        success = True
        
        print(f"\\n✅ [成功完成] 耗時: {elapsed_time:.2f}s | 吞吐量: {throughput:.2f} tokens/s")
        print(f"📏 [CUDA Driver API 實測 GPU 顯存]: {measured_physical_vram_gb:.2f} GB")
        
    except Exception as e:
        error_msg = str(e)
        print(f"\\n❌ [推論失敗/OOM/崩潰] 原因: {error_msg}")
        
    result_data = {
        "mode": mode,
        "enable_chunked_prefill": use_chunked,
        "success": success,
        "error_msg": error_msg,
        "elapsed_time_s": round(elapsed_time, 2),
        "throughput_tps": round(throughput, 2),
        "model_vram_gb": model_vram,
        "gpu_reserved_limit_gb": gpu_reserved_vram,
        "measured_physical_vram_gb": measured_physical_vram_gb,
        "est_kv_demand_gib": 3.30
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