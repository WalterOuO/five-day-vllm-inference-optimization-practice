import json
import os
import argparse
import time
import torch
import yaml
from transformers import AutoConfig
from vllm import LLM, SamplingParams

def get_physical_vram_gb():
    """透過 CUDA Driver API 獲取當前 GPU 已使用的真實物理 VRAM (GB)"""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return (total_bytes - free_bytes) / (1024**3)

def compute_model_weight_vram(model_id):
    """動態讀取 AutoConfig 計算模型權重顯存 (GB)"""
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

    # FP16 = 2.0 Bytes per parameter
    model_weight_bytes = total_params * 2.0
    return round(model_weight_bytes / (1024**3), 2), config

def compute_kv_cache_bytes_per_token(config):
    """計算單個 Token 的 KV Cache 位元組數 (FP16)"""
    num_layers = getattr(config, "num_hidden_layers", 36)
    num_heads = getattr(config, "num_attention_heads", 16)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    hidden_size = getattr(config, "hidden_size", 2048)
    head_dim = hidden_size // num_heads

    # Key + Value Tensor (FP16 = 2 Bytes)
    return 2 * num_layers * num_kv_heads * head_dim * 2.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "chunked_prefill"], required=True)
    parser.add_argument("--config", default="configs/day2_chunked_prefill.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    model_id = config_yaml["model"]
    model_weight_gb, model_config = compute_model_weight_vram(model_id)
    kv_bytes_per_token = compute_kv_cache_bytes_per_token(model_config)

    prompts_path = config_yaml["prompts_json_path"]
    if not os.path.exists(prompts_path):
        prompts_path = "/data/prompts/benchmark_prompts.json"

    with open(prompts_path, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)

    prompts_list = [p if isinstance(p, str) else p.get("prompt", str(p)) for p in raw_prompts]

    def build_batch(gradient_k, concurrency=4):
        batch = []
        for i in range(concurrency):
            start_idx = (i * gradient_k) % len(prompts_list)
            chunk = [prompts_list[(start_idx + j) % len(prompts_list)] for j in range(gradient_k)]
            batch.append("\\n\\n".join(chunk))
        return batch

    gradients = config_yaml["gradients_k"]
    enable_chunked = (args.mode == "chunked_prefill")
    max_chunk_tokens = config_yaml["max_num_batched_tokens"] if enable_chunked else None

    print(f"🚀 [Day 2 Real Test] 模式: {args.mode} | Chunked Prefill: {enable_chunked} | 長度梯度: {gradients}")

    llm = LLM(
        model=model_id,
        dtype="float16",
        max_model_len=20480,
        gpu_memory_utilization=0.85,
        enable_chunked_prefill=enable_chunked,
        max_num_batched_tokens=max_chunk_tokens,
        enforce_eager=True
    )

    real_physical_vram = get_physical_vram_gb()
    sampling_params = SamplingParams(temperature=0.7, max_tokens=config_yaml["generation_tokens"])
    metrics = []

    for grad in gradients:
        torch.cuda.empty_cache()
        batch_prompts = build_batch(grad, concurrency=config_yaml["concurrency"])

        start_time = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        total_prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
        total_gen_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        total_batch_tokens = total_prompt_tokens + total_gen_tokens

        # 1. 計算實際 KV Cache 佔用 (GB)
        active_kv_vram_gb = (total_batch_tokens * kv_bytes_per_token) / (1024**3)
        total_active_vram_gb = model_weight_gb + active_kv_vram_gb

        # 2. 計算 Prefill 階段單步 Batched Tokens 峰值 (Tokens)
        if enable_chunked:
            prefill_step_max_tokens = min(total_prompt_tokens, config_yaml["max_num_batched_tokens"])
        else:
            prefill_step_max_tokens = total_prompt_tokens

        decode_tps = total_gen_tokens / total_time if total_time > 0 else 0
        itl_ms = (total_time / total_gen_tokens * 1000) if total_gen_tokens > 0 else 0

        res = {
            "gradient": f"{grad}K",
            "prompt_len_k": grad,
            "model_weight_vram_gb": model_weight_gb,
            "active_kv_vram_gb": round(active_kv_vram_gb, 4),
            "total_active_vram_gb": round(total_active_vram_gb, 4),
            "total_physical_vram_gb": round(real_physical_vram, 2),
            "prefill_step_max_tokens": prefill_step_max_tokens,
            "decode_tps": round(decode_tps, 2),
            "itl_ms": round(itl_ms, 2),
            "total_prompt_tokens": total_prompt_tokens,
            "total_gen_tokens": total_gen_tokens,
            "latency_sec": round(total_time, 2)
        }
        metrics.append(res)
        print(f"  └─ 梯度 {grad}K | Prefill 單步 Batch 峰值: {prefill_step_max_tokens} Tokens | KV VRAM: {active_kv_vram_gb:.4f} GB | Decode TPS: {decode_tps:.2f} | Latency: {total_time:.2f}s")

    out_file = f"results/raw_logs/day2_{args.mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "metrics": metrics}, f, indent=2, ensure_ascii=False)

    print(f"✅ 實測完成，數據已寫入 {out_file}")

if __name__ == "__main__":
    main()