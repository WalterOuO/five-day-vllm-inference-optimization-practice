import os
import sys
import time
import json
import yaml
import re
import subprocess
import requests
import argparse
import torch

def print_log_tail(log_path, lines=30):
    """印出日誌檔案最後幾行以方便除錯"""
    if os.path.exists(log_path):
        print(f"\n===== 📄 vLLM 啟動日誌摘要 (最後 {lines} 行: {log_path}) =====")
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
            for line in log_lines[-lines:]:
                print(line.rstrip())
        print("========================================================\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "apc"], required=True, help="baseline: 無前綴快取, apc: 開啟 Automatic Prefix Caching")
    parser.add_argument("--config", default="configs/day3_apc.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mode = args.mode
    model = cfg["model"]
    port = cfg.get("port", 8000)
    gpu_util = cfg.get("gpu_memory_utilization", 0.80)
    max_len = cfg.get("max_model_len", 4096)
    req_rate = cfg.get("request_rate", 2.0)

    # 清理 PyTorch 殘留顯存，防止 OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. 讀取 benchmark_prompts.json 並建立 Day 3 測試數據集
    prompt_file = "data/prompts/benchmark_prompts.json"
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"找不到 {prompt_file}，請確認是否已生成！")

    with open(prompt_file, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)

    # 拼接前 2 個 Prompt 形成約 2K Tokens 的 System Prompt
    system_prompt = raw_prompts[0] + "\n\n" + raw_prompts[1]

    # 取第 3~7 個 Prompt 的前 300 字元（約 50~100 Tokens）作為 5 個不同的 User Prompt
    requests_data = []
    for i in range(5):
        user_q = raw_prompts[2 + i][:300]
        full_prompt = f"[SYSTEM PROMPT]\n{system_prompt}\n\n[USER QUESTION]\n{user_q}"
        requests_data.append({
            "prompt": full_prompt
        })

    # 關鍵修復：vllm bench serve 的 CustomDataset 要求必須是 .jsonl 格式 (逐行 JSON)
    dataset_path = "data/prompts/day3_dataset.jsonl"
    os.makedirs("data/prompts", exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for req in requests_data:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"✅ 已成功構建 5 個請求數據集 ({dataset_path})：共用 2K System Prompt + 5 個獨立 User Questions")

    # 2. 準備日誌輸出檔案
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = f"{log_dir}/vllm_server_{mode}.log"
    log_file = open(log_file_path, "w", encoding="utf-8")

    # 建立 vLLM 啟動指令
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_len),
        "--dtype", "float16",
        "--trust-remote-code"
    ]

    if mode == "baseline":
        server_cmd.append("--no-enable-prefix-caching")
    else:
        server_cmd.append("--enable-prefix-caching")

    flag_used = "--no-enable-prefix-caching" if mode == "baseline" else "--enable-prefix-caching"
    print(f"🚀 啟動 vLLM Server ({mode.upper()} 模式, 旗標: {flag_used})...")
    print(f"📝 伺服器啟動日誌同步寫入: {log_file_path}")

    # 以背景子行程啟動，並將 stdout/stderr 導向日誌檔
    server_proc = subprocess.Popen(server_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # 等待 Server 健康檢查通關 (最多等待 240 秒 / 4 分鐘)
    health_url = f"http://localhost:{port}/health"
    start_wait = time.time()
    server_ready = False
    
    while time.time() - start_wait < 240:
        # 檢查子行程是否提前崩潰退出
        poll_code = server_proc.poll()
        if poll_code is not None:
            log_file.close()
            print(f"\n❌ vLLM Server 進程已異常崩潰退出 (Exit Code: {poll_code})！")
            print_log_tail(log_file_path, lines=40)
            raise RuntimeError(f"vLLM Server 啟動失敗，請檢查日誌 {log_file_path}")

        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                server_ready = True
                print("✅ vLLM Server 已順利啟動完成！")
                break
        except Exception:
            pass
        time.sleep(4)

    if not server_ready:
        server_proc.kill()
        log_file.close()
        print("\n❌ vLLM Server 啟動超時 (已等待 240 秒)！")
        print_log_tail(log_file_path, lines=40)
        raise RuntimeError(f"vLLM Server 啟動超時，詳情請參閱 {log_file_path}")

    # 3. 執行官方 vllm bench serve 線上測試 (指向 .jsonl 檔案)
    bench_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.bench_serve",
        "--base-url", f"http://localhost:{port}",
        "--model", model,
        "--dataset-name", "custom",
        "--dataset-path", dataset_path,
        "--num-prompts", "5",
        "--request-rate", str(req_rate)
    ]
    
    alt_bench_cmd = [
        "vllm", "bench", "serve",
        "--base-url", f"http://localhost:{port}",
        "--model", model,
        "--dataset-name", "custom",
        "--dataset-path", dataset_path,
        "--num-prompts", "5",
        "--request-rate", str(req_rate)
    ]

    print("\n📢 執行 vLLM 官方 'vllm bench serve' 線上測試...")
    try:
        bench_res = subprocess.run(bench_cmd, capture_output=True, text=True)
        if bench_res.returncode != 0:
            bench_res = subprocess.run(alt_bench_cmd, capture_output=True, text=True)
    except Exception as e:
        bench_res = subprocess.run(alt_bench_cmd, capture_output=True, text=True)

    # 完整印出官方壓測報告 stdout
    stdout_text = bench_res.stdout
    print("\n==================== vLLM Bench Serve 官方線上測試報告 ====================")
    print(stdout_text)
    if bench_res.stderr:
        print("---------- Stderr Warning/Error ----------")
        print(bench_res.stderr)
    print("==========================================================================")

    # 4. 延遲 2 秒確保 Prometheus Metrics 完成異步更新
    print("⏳ 等待 2 秒以利 Prometheus Metrics 完成異步數據刷新...")
    time.sleep(2)

    # 5. 從 /metrics 端點採集真實 Prefix Cache Hit Rate
    metrics_url = f"http://localhost:{port}/metrics"
    cache_hit_rate = 0.0
    hits, queries = 0.0, 0.0
    try:
        m_resp = requests.get(metrics_url, timeout=5)
        if m_resp.status_code == 200:
            for line in m_resp.text.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "vllm:prefix_cache_hits_total" in line:
                    hits = float(line.split()[-1])
                elif "vllm:prefix_cache_queries_total" in line:
                    queries = float(line.split()[-1])
                elif "vllm:gpu_prefix_cache_hit_rate" in line:
                    cache_hit_rate = float(line.split()[-1]) * 100.0
            if queries > 0:
                cache_hit_rate = (hits / queries) * 100.0
    except Exception as e:
        print(f"⚠️ 讀取 /metrics 發生異常: {e}")

    # 正則萃取 TTFT 指標
    p99_match = (
        re.search(r'(?:P99|99th|p99).*?TTFT.*?:?\s*([\d\.]+)', stdout_text, re.IGNORECASE)
        or re.search(r'TTFT.*?P99.*?:?\s*([\d\.]+)', stdout_text, re.IGNORECASE)
    )
    mean_match = (
        re.search(r'(?:Mean|Average|Avg).*?TTFT.*?:?\s*([\d\.]+)', stdout_text, re.IGNORECASE)
        or re.search(r'TTFT.*?(?:Mean|Average|Avg).*?:?\s*([\d\.]+)', stdout_text, re.IGNORECASE)
    )

    ttft_p99 = float(p99_match.group(1)) if p99_match else None
    ttft_avg = float(mean_match.group(1)) if mean_match else None

    summary = {
        "mode": mode,
        "model": model,
        "num_requests": 5,
        "ttft_p99_ms": ttft_p99,
        "ttft_avg_ms": ttft_avg,
        "cache_hits": int(hits),
        "cache_queries": int(queries),
        "cache_hit_rate_pct": round(cache_hit_rate, 2),
        "raw_bench_stdout": stdout_text
    }

    # 6. 關閉 vLLM 伺服器與日誌檔
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    log_file.close()

    # 7. 寫入 JSON 日誌
    os.makedirs("results/raw_logs", exist_ok=True)
    out_file = f"results/raw_logs/day3_{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ {mode.upper()} 實驗完成！")
    print(f"🎯 真實 Cache Hit Rate (來自 /metrics): {cache_hit_rate:.2f}% (Hits: {int(hits)}/{int(queries)})")
    print(f"📄 結果已存至 {out_file}")

if __name__ == "__main__":
    main()