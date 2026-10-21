import argparse
import time
import json
import torch
import yaml
from vllm import LLM, SamplingParams

def run_experiment(mode, config_path):
  with open(config_path, "r", encoding="utf-8") as f:
      config = yaml.safe_load(f)
  
  base_cfg = config["base"]
  exp_cfg = config[mode]
  
  print(f"========== 啟動 Day 1 實驗: {mode.upper()} ==========")
  print(f"模型: {exp_cfg['model_id']}")
  print(f"Prompt 通用資料庫來源: {base_cfg['prompt_file']}")
  
  # 讀取包含 32 個獨立 1K Prompt 的通用資料庫
  with open(base_cfg['prompt_file'], "r", encoding="utf-8") as f:
      prompts = json.load(f)
  
  concurrency = len(prompts)
  print(f"🚀 成功載入 {concurrency} 個內容獨立的 Prompts (高併發壓測)")
  print(f"生成上限 (Max Gen Tokens): {base_cfg['max_gen_tokens']}")
  
  # 紀錄初始顯存狀態
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  vram_before_load = torch.cuda.memory_allocated() / (1024**3)
  
  # 載入 vLLM 模型
  llm = LLM(
      model=exp_cfg['model_id'],
      quantization=exp_cfg['quantization'],
      dtype=exp_cfg['dtype'],
      gpu_memory_utilization=base_cfg['gpu_memory_utilization'],
      trust_remote_code=True
  )
  
  vram_after_load = torch.cuda.memory_allocated() / (1024**3)
  static_vram = vram_after_load - vram_before_load
  print(f"📊 靜態顯存佔用 (Static VRAM): {static_vram:.2f} GB")
  
  sampling_params = SamplingParams(
      temperature=0.0,
      max_tokens=base_cfg['max_gen_tokens']
  )
  
  # 發送 32 個不同請求進行高併發推論壓測
  start_time = time.perf_counter()
  outputs = llm.generate(prompts, sampling_params)
  end_time = time.perf_counter()
  
  total_time = end_time - start_time
  total_gen_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
  system_tps = total_gen_tokens / total_time
  
  peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
  
  print(f"⏱️ 總執行時間: {total_time:.2f} 秒")
  print(f"🚀 總生成 Tokens: {total_gen_tokens}")
  print(f"🎯 系統總吞吐量 (System Total TPS): {system_tps:.2f} tokens/sec")
  print(f"📉 Peak VRAM (峰值顯存): {peak_vram:.2f} GB")
  
  result = {
      "mode": mode,
      "model": exp_cfg['model_id'],
      "prompt_file": base_cfg['prompt_file'],
      "concurrency": concurrency,
      "static_vram_gb": round(static_vram, 2),
      "peak_vram_gb": round(peak_vram, 2),
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