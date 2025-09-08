import platform
import psutil
import torch
import time, requests, json, statistics

results = {}

# === System Specs ===
results["system"] = {
    "os": f"{platform.system()} {platform.version()}",
    "cpu": platform.processor(),
    "cores": psutil.cpu_count(logical=False),
    "threads": psutil.cpu_count(logical=True),
    "ram_gb": round(psutil.virtual_memory().total / 1e9, 2)
}

# === PyTorch & GPU ===
gpu_info = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available()
}
if torch.cuda.is_available():
    gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
    gpu_info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
results["gpu"] = gpu_info

# === Network Latency (Binance) ===
url = "https://api.binance.com/api/v3/ping"
latencies = []
for _ in range(10):
    start = time.time()
    requests.get(url)
    latencies.append((time.time() - start) * 1000)  # ms

results["latency"] = {
    "samples_ms": [round(l, 2) for l in latencies],
    "avg_ms": round(statistics.mean(latencies), 2),
    "min_ms": round(min(latencies), 2),
    "max_ms": round(max(latencies), 2),
    "stdev_ms": round(statistics.pstdev(latencies), 2)
}

# === Save JSON ===
with open("diagnostics.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Diagnostics written to diagnostics.json")
