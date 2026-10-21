import argparse
import time
import json
import torch
import yaml
from transformers import AutoConfig
from vllm import LLM, SamplingParams

def get_physical_vram_gb():
    """透過 CUDA Driver API 獲取當前 GPU 已使用的真實物理 VRAM (GB)"""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return used_bytes / (1024**3)

def compute_model_weight_vram_dynamically(model_id, quantization):
    """
    完全不寫死數字！
    動態讀取 Hugging Face Model Config，精確計算模型參數量與理論權重佔用 (GB)
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # 讀取 Transformer 架構核心參數
    vocab_size = getattr(config, "vocab_size", 151936)
    hidden_size = getattr(config, "hidden_size", 2048)
    intermediate_size = getattr(config, "intermediate_size", 11008)
    num_layers = getattr(config, "num_hidden_layers", 36)
    num_heads = getattr(config, "num_attention_heads", 16)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    
    # 計算單個 Layer 的參數數量 (Attention + MLP)
    head_dim = hidden_size // num_heads
    q_proj = hidden_size * (num_heads * head_dim)
    k_proj = hidden_size * (num_kv_heads * head_dim)
    v_proj = hidden_size * (num_kv_heads * head_dim)
    out_proj = (num_heads * head_dim) * hidden_size
    attn_params = q_proj + k_proj + v_proj + out_proj
    
    mlp_params = 3 * hidden_size * intermediate_size # SwiGLU / Gate-Up-Down
    layer_params = attn_params + mlp_params
    
    # 總參數量 = 詞表 Embedding + 所有 Layers + LM Head
    total_params = (vocab_size * hidden_size) + (num_layers * layer_params) + (vocab_size * hidden_size)
    
    # 根據量化類型決定每個參數佔用的位元組 (Bytes per Parameter)
    if quantization == "awq":
        bytes_per_param = 0.5  # 4-bit INT4
        quant_overhead_factor = 1.15 # AWQ 包含 FP16 Scales 與 Zero-points 的額外開銷
    else:
        bytes_per_param = 2.0  # 16-bit FP16 / BF16
        quant_overhead_factor = 1.0
        
    model_weight_bytes = total_params * bytes_per_param * quant_overhead_factor
    model_weight_gb = model_weight_bytes / (1024**3)
    
    return round(model_weight_gb, 2), total_params

def run_experiment(mode, config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    base_cfg = config["base"]
    exp_cfg = config[mode]
    
    print(f"========== 啟動 Day 1 實驗: {mode.upper()} ==========")
    print(f"模型 ID: {exp_cfg['model_id']}")
    print(f"Prompt 來源: {base_cfg['prompt_file']}")
    
    # 1. 動態計算模型權重體積 (非寫死)
    model_weight_vram, total_params = compute_model_weight_vram_dynamically(
        exp_cfg['model_id'], 
        exp_cfg['quantization']
    )
    print(f"🔍 [動態模型結構解析] 總參數量: {total_params / 1e9:.2f} B ({total_params:,} params)")
    print(f"🔍 [動態權重計算] 理論模型權重佔用: {model_weight_vram:.2f} GB")
    
    with open(base_cfg['prompt_file'], "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    concurrency = len(prompts)
    print(f"🚀 成功載入 {concurrency} 個內容獨立的 Prompts (高併發壓測)")
    
    torch.cuda.empty_cache()
    
    # 2. 載入 vLLM 模型
    llm = LLM(
        model=exp_cfg['model_id'],
        quantization=exp_cfg['quantization'],
        dtype=exp_cfg['dtype'],
        gpu_memory_utilization=base_cfg['gpu_memory_utilization'],
        trust_remote_code=True
    )
    
    # 3. 獲取 vLLM 預留的總物理顯存 (約為 GPU 總量的 85%)
    total_reserved_vram = get_physical_vram_gb()
    
    # 4. 動態推導可用的 KV Cache 記憶體池
    kv_cache_pool_vram = max(0.0, total_reserved_vram - model_weight_vram)
    
    print(f"\n📊 [真實動態顯存結構分析]")
    print(f"   ├── 1. 動態計算模型權重 (Model Weight VRAM): {model_weight_vram:.2f} GB")
    print(f"   ├── 2. 剩餘可用 KV Cache 記憶體池 (KV Cache Pool): {kv_cache_pool_vram:.2f} GB")
    print(f"   └── 3. vLLM 總預留物理顯存 (Total Reserved VRAM): {total_reserved_vram:.2f} GB")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=base_cfg['max_gen_tokens']
    )
    
    # 發送高併發請求
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_gen_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    system_tps = total_gen_tokens / total_time
    
    print(f"\n⏱️ 總執行時間: {total_time:.2f} 秒")
    print(f"🚀 總生成 Tokens: {total_gen_tokens}")
    print(f"🎯 系統總吞吐量 (System Total TPS): {system_tps:.2f} tokens/sec")
    
    result = {
        "mode": mode,
        "model": exp_cfg['model_id'],
        "prompt_file": base_cfg['prompt_file'],
        "concurrency": concurrency,
        "total_parameters_count": total_params,
        "model_weight_vram_gb": round(model_weight_vram, 2),
        "kv_cache_pool_vram_gb": round(kv_cache_pool_vram, 2),
        "total_reserved_vram_gb": round(total_reserved_vram, 2),
        "total_time_sec": round(total_time, 2),
        "total_gen_tokens": total_gen_tokens,
        "system_tps": round(system_tps, 2)
    }
    
    output_file = f"results/raw_logs/day1_{mode}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"💾 數據結果已儲存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "awq"])
    parser.add_argument("--config", type=str, default="configs/day1_awq.yaml")
    args = parser.parse_args()
    
    run_experiment(args.mode, args.config)