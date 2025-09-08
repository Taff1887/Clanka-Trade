#!/usr/bin/env python3
"""
hft_env_probe.py — Environment & latency probe for Binance Testnets (Spot & Futures)

What it does:
- Prints detailed system specs (OS, CPU, RAM, disk, Python, PyTorch, CUDA/GPU)
- Checks high-resolution timer precision & system time offset (via NTP if available)
- Measures TCP, HTTP(S), and optional WebSocket handshake latency to Binance *testnet* endpoints
- Saves a JSON report + raw samples to files for sharing: hft_env_report.json (summary), hft_env_raw_samples.json

Usage:
  python hft_env_probe.py               # test both Spot & Futures testnets
  python hft_env_probe.py --env spot    # Spot testnet only
  python hft_env_probe.py --env futures # Futures testnet only
  python hft_env_probe.py --samples 10  # change sample count

Recommended (Windows PowerShell):
  py -3 hft_env_probe.py --env all --samples 8

Dependencies (auto-detected; optional where noted):
  pip install psutil requests
  pip install websockets ntplib  # optional, enables ws latency & NTP offset
  # PyTorch is optional; if installed, GPU/CUDA info is included

Author: ChatGPT (GPT-5 Thinking)
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import json
import math
import os
import platform
import shutil
import socket
import ssl
import statistics as stats
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# ---- Optional imports ----
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

try:
    import websockets  # type: ignore
except Exception:
    websockets = None  # type: ignore

try:
    import ntplib  # type: ignore
except Exception:
    ntplib = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def _run(cmd: List[str], timeout: float = 5.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True, shell=False)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def gather_system_info() -> Dict:
    info = {
        "collected_at_utc": dt.datetime.utcnow().isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "uname": dict(zip(("system","node","release","version","machine","processor"), platform.uname())),
        "perf_counter_resolution_ns": int(time.get_clock_info("perf_counter").resolution * 1e9),
    }

    # CPU, RAM, Disk
    if psutil:
        try:
            info["cpu_logical"] = psutil.cpu_count(logical=True)
            info["cpu_physical"] = psutil.cpu_count(logical=False)
            freq = psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz"] = {"current": freq.current, "min": freq.min, "max": freq.max}
            vm = psutil.virtual_memory()
            info["ram_total"] = _human_bytes(vm.total)
            du = psutil.disk_usage(os.path.abspath(os.sep))
            info["disk_total"] = _human_bytes(du.total)
        except Exception as e:
            info["psutil_error"] = str(e)
    else:
        info["psutil_missing"] = True

    # PyTorch / CUDA / GPU
    torch_info = {}
    if torch:
        torch_info["torch_version"] = getattr(torch, "__version__", "unknown")
        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        torch_info["cuda_version"] = getattr(torch.version, "cuda", None)
        torch_info["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: None)()
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                gpus.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_mem_GB": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                })
            torch_info["gpus"] = gpus
    else:
        torch_info["torch_missing"] = True

    # nvidia-smi fallback if torch not available
    if (not torch) or (torch and not torch.cuda.is_available()):
        code, out, _ = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        if code == 0 and out:
            lines = [x.strip() for x in out.splitlines() if x.strip()]
            torch_info["nvidia_smi"] = lines

    info["pytorch_cuda"] = torch_info
    return info


def ntp_offset_ms() -> Optional[float]:
    # Positive offset means local clock is **behind** server by that many ms
    # (per ntplib docs: offset = ((t2 - t1) + (t3 - t4)) / 2)
    if not ntplib:
        return None
    client = ntplib.NTPClient()
    for host in ["time.google.com", "pool.ntp.org"]:
        try:
            r = client.request(host, version=3, timeout=3)
            return r.offset * 1000.0
        except Exception:
            continue
    return None


def windows_time_service_status() -> Optional[str]:
    if platform.system().lower() != "windows":
        return None
    code, out, err = _run(["w32tm", "/query", "/status"], timeout=4.0)
    if code == 0 and out:
        return out
    return err or None


def tcp_latency(host: str, port: int, samples: int = 6, timeout: float = 2.0) -> Dict:
    # TCP connect latency (SYN -> 3-way handshake)
    vals = []
    for _ in range(samples):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout) as s:
                s.settimeout(timeout)
        except Exception:
            vals.append(math.nan)
        else:
            vals.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(0.05)
    clean = [v for v in vals if not math.isnan(v)]
    summary = {
        "host": host, "port": port, "samples": samples,
        "ms_p50": round(stats.median(clean), 3) if clean else None,
        "ms_p90": round(_percentile(clean, 90), 3) if clean else None,
        "ms_min": round(min(clean), 3) if clean else None,
        "ms_max": round(max(clean), 3) if clean else None,
        "loss": round(100.0 * (len(vals) - len(clean)) / len(vals), 1) if vals else None,
        "raw_ms": [None if math.isnan(v) else round(v, 3) for v in vals],
    }
    return summary


def https_latency(url: str, samples: int = 4, timeout: float = 3.5) -> Dict:
    if not requests:
        return {"url": url, "error": "requests not installed"}
    vals = []
    for _ in range(samples):
        t0 = time.perf_counter()
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
        except Exception:
            vals.append(math.nan)
        else:
            vals.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(0.05)
    clean = [v for v in vals if not math.isnan(v)]
    return {
        "url": url, "samples": samples,
        "ms_p50": round(stats.median(clean), 3) if clean else None,
        "ms_p90": round(_percentile(clean, 90), 3) if clean else None,
        "ms_min": round(min(clean), 3) if clean else None,
        "ms_max": round(max(clean), 3) if clean else None,
        "loss": round(100.0 * (len(vals) - len(clean)) / len(vals), 1) if vals else None,
        "raw_ms": [None if math.isnan(v) else round(v, 3) for v in vals],
    }


async def _ws_connect_once(url: str, timeout: float = 4.0) -> Optional[float]:
    if not websockets:
        return None
    t0 = time.perf_counter()
    try:
        async with websockets.connect(url, max_queue=None, open_timeout=timeout, ping_timeout=timeout) as ws:
            # Measure a ping/pong round-trip as an approximation to ongoing heartbeat latency
            t1 = time.perf_counter()
            try:
                await ws.ping()
                await asyncio.wait_for(ws.pong_waiter(), timeout=timeout)
                t2 = time.perf_counter()
                return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0  # handshake_ms, pingpong_ms
            except Exception:
                return (t1 - t0) * 1000.0, None
    except Exception:
        return None


def _percentile(arr: List[float], p: float) -> float:
    if not arr:
        return float("nan")
    arr2 = sorted(arr)
    k = (len(arr2) - 1) * (p / 100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return arr2[int(k)]
    return arr2[f] + (arr2[c] - arr2[f]) * (k - f)


async def websocket_latency(url: str, samples: int = 3, timeout: float = 4.0) -> Dict:
    if not websockets:
        return {"url": url, "error": "websockets not installed"}
    handshakes = []
    pingpongs = []
    for _ in range(samples):
        res = await _ws_connect_once(url, timeout=timeout)
        if not res:
            handshakes.append(math.nan); pingpongs.append(math.nan)
        else:
            hs, pp = res
            handshakes.append(hs if hs is not None else math.nan)
            pingpongs.append(pp if pp is not None else math.nan)
        await asyncio.sleep(0.05)

    def summarize(vals: List[Optional[float]]):
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return {
            "p50": round(stats.median(clean), 3) if clean else None,
            "p90": round(_percentile(clean, 90), 3) if clean else None,
            "min": round(min(clean), 3) if clean else None,
            "max": round(max(clean), 3) if clean else None,
            "loss": round(100.0 * (len(vals) - len(clean)) / len(vals), 1) if vals else None,
            "raw": [None if (v is None or math.isnan(v)) else round(v, 3) for v in vals]
        }

    return {
        "url": url,
        "handshake_ms": summarize(handshakes),
        "pingpong_ms": summarize(pingpongs)
    }


def binance_testnet_targets(env: str) -> Dict[str, Dict[str, List[str]]]:
    # Based on Binance Open Platform docs
    return {
        "spot": {
            "tcp": [
                "testnet.binance.vision:443",
                "stream.testnet.binance.vision:443",
                "stream.testnet.binance.vision:9443",
                "ws-api.testnet.binance.vision:443",
                "ws-api.testnet.binance.vision:9443",
            ],
            "https": [
                "https://testnet.binance.vision/api/v3/time",
                "https://testnet.binance.vision/api/v3/exchangeInfo",
            ],
            "wss": [
                "wss://stream.testnet.binance.vision/ws",
                "wss://ws-api.testnet.binance.vision/ws-api/v3",
            ]
        },
        "futures": {
            "tcp": [
                "testnet.binancefuture.com:443",
                "fstream.binancefuture.com:443",
                "dstream.binancefuture.com:443",
            ],
            "https": [
                "https://testnet.binancefuture.com/fapi/v1/time",
                "https://testnet.binancefuture.com/fapi/v1/exchangeInfo",
            ],
            "wss": [
                "wss://fstream.binancefuture.com/ws",
                "wss://dstream.binancefuture.com/ws",
            ]
        }
    }[env]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["spot", "futures", "all"], default="all", help="Which testnet(s) to probe")
    ap.add_argument("--samples", type=int, default=6, help="Sample count per check (TCP/HTTP/WS)")
    args = ap.parse_args()

    summary = {
        "system": gather_system_info(),
        "time_sync": {
            "ntp_offset_ms": ntp_offset_ms(),
            "windows_time_service": windows_time_service_status(),
        },
        "latency": {},
    }
    raw_samples = {"tcp": [], "https": [], "wss": []}

    envs = ["spot", "futures"] if args.env == "all" else [args.env]
    for env in envs:
        tgt = binance_testnet_targets(env)
        summary["latency"][env] = {"tcp": [], "https": [], "wss": []}

        # TCP
        for hp in tgt["tcp"]:
            host, port = hp.split(":")
            res = tcp_latency(host, int(port), samples=args.samples)
            summary["latency"][env]["tcp"].append(res)
            raw_samples["tcp"].append({"env": env, **res})

        # HTTPS
        for url in tgt["https"]:
            res = https_latency(url, samples=max(3, args.samples//2))
            summary["latency"][env]["https"].append(res)
            raw_samples["https"].append({"env": env, **res})

        # WebSocket
        if websockets:
            loop = asyncio.get_event_loop()
            for url in tgt["wss"]:
                res = loop.run_until_complete(websocket_latency(url, samples=max(3, args.samples//2)))
                summary["latency"][env]["wss"].append(res)
                raw_samples["wss"].append({"env": env, **res})
        else:
            summary["latency"][env]["wss_missing_websockets_lib"] = True

    # Write reports
    out_json = "hft_env_report.json"
    raw_json = "hft_env_raw_samples.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(raw_samples, f, indent=2)

    # Pretty print short summary
    print("="*72)
    print("HFT Environment Probe — SUMMARY")
    print("="*72)
    sysinfo = summary["system"]
    print(f"Collected at (UTC): {sysinfo['collected_at_utc']}")
    print(f"OS: {sysinfo['platform']}")
    print(f"Python: {sysinfo['python']}")
    if 'cpu_logical' in sysinfo:
        print(f"CPU cores (logical/physical): {sysinfo.get('cpu_logical')} / {sysinfo.get('cpu_physical')}")
    if 'ram_total' in sysinfo:
        print(f"RAM total: {sysinfo['ram_total']}   Disk total: {sysinfo.get('disk_total', 'n/a')}")
    torch_cuda = sysinfo.get("pytorch_cuda", {})
    if not torch_cuda.get("torch_missing"):
        print(f"PyTorch: {torch_cuda.get('torch_version')}  CUDA: {torch_cuda.get('cuda_version')}  cuDNN: {torch_cuda.get('cudnn_version')}")
        if torch_cuda.get("gpus"):
            for g in torch_cuda["gpus"]:
                print(f" GPU[{g['index']}]: {g['name']}  {g['total_mem_GB']} GB  CC {g['capability']}")
    else:
        print("PyTorch not installed — GPU/CUDA info skipped")

    # Time sync
    print("-"*72)
    ntp = summary["time_sync"].get("ntp_offset_ms")
    print(f"NTP offset (ms): {round(ntp,2) if ntp is not None else 'ntplib not installed or failed'}")
    if summary["time_sync"].get("windows_time_service"):
        print("Windows time service:\n" + summary["time_sync"]["windows_time_service"])

    # Latency digest
    print("-"*72)
    for env in envs:
        print(f"[{env.upper()} TESTNET]")
        for item in summary["latency"][env]["tcp"]:
            print(f" TCP {item['host']}:{item['port']}  p50={item['ms_p50']} ms  p90={item['ms_p90']} ms  loss={item['loss']}%")
        for item in summary["latency"][env]["https"]:
            print(f" HTTPS {item['url']}  p50={item['ms_p50']} ms  p90={item['ms_p90']} ms  loss={item['loss']}%")
        wss = summary["latency"][env].get("wss", [])
        if isinstance(wss, list):
            for item in wss:
                hs = item['handshake_ms']; pp = item['pingpong_ms']
                print(f" WSS {item['url']}  hs_p50={hs['p50']} ms  pong_p50={pp['p50']} ms  loss_hs={hs['loss']}%")
        else:
            print(" WSS: websockets library missing — skipped")

    print("-"*72)
    print(f"Wrote: {out_json} (summary)")
    print(f"Wrote: {raw_json} (raw samples)")


if __name__ == "__main__":
    main()
