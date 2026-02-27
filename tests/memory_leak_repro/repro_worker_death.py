#!/usr/bin/env python3
"""
Memory Leak Reproduction — Demonstrates workers dying from unbounded memory growth.

This script reproduces the production issue where litellm proxy workers grow from
~300MB to multi-GB and then crash, because PrismaClient.spend_log_transactions is
an unbounded list that never drains when the DB is unreachable.

Strategy:
  1. Start a fake OpenAI server (in-thread)
  2. Start the litellm proxy via a wrapper module that:
     a) Sets RLIMIT_AS to cap worker memory at ~350MB
     b) Installs a middleware that simulates the spend_log_transactions leak
        (deepcopy a ~3KB payload per request, never drained)
  3. Blast requests with high concurrency
  4. Watch workers grow in memory and die when they hit the cap
  5. Watch uvicorn respawn them, and the cycle repeats

Usage:
    cd /workspace
    poetry run python tests/memory_leak_repro/repro_worker_death.py
"""

import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from threading import Thread

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAKE_OPENAI_PORT = 18080
PROXY_PORT = 4001
NUM_WORKERS = 2
WORKER_MEM_CAP_MB = 800  # VmSize limit; baseline ~550MB after warmup, leaves ~250MB before crash
BLAST_CONCURRENCY = 30
BLAST_TOTAL = 200000
POLL_INTERVAL = 2
MAX_DURATION = 300  # 5 minutes max

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def ts():
    return datetime.now().strftime("%H:%M:%S")


def wait_for_port(port, host="127.0.0.1", timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def get_child_pids(parent_pid):
    children = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat") as f:
                    stat = f.read()
                parts = stat.split(")")
                if len(parts) >= 2:
                    fields = parts[-1].split()
                    ppid = int(fields[1])
                    if ppid == parent_pid:
                        children.append(pid)
            except (FileNotFoundError, PermissionError, ValueError, IndexError):
                continue
    except Exception:
        pass
    return children


def get_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# 1. Fake OpenAI server
# ---------------------------------------------------------------------------
def start_fake_openai():
    import uvicorn
    sys.path.insert(0, WORKSPACE)
    from tests.memory_leak_repro.fake_openai_server import app as fake_app

    thread = Thread(
        target=lambda: uvicorn.run(
            fake_app, host="127.0.0.1", port=FAKE_OPENAI_PORT,
            log_level="error", access_log=False,
        ),
        daemon=True,
    )
    thread.start()
    if not wait_for_port(FAKE_OPENAI_PORT, timeout=10):
        print(f"[{ts()}] FATAL: fake OpenAI server did not start")
        sys.exit(1)
    print(f"[{ts()}] ✓ Fake OpenAI server ready on :{FAKE_OPENAI_PORT}")


# ---------------------------------------------------------------------------
# 2. Write proxy config
# ---------------------------------------------------------------------------
def write_proxy_config():
    import yaml
    config = {
        "model_list": [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_base": f"http://127.0.0.1:{FAKE_OPENAI_PORT}/v1",
                    "api_key": "sk-fake",
                },
            }
        ],
        "general_settings": {
            "master_key": "sk-test-master-key",
            "disable_reset_budget": True,
        },
    }
    path = os.path.join(SCRIPT_DIR, "_repro_config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ---------------------------------------------------------------------------
# 3. Start proxy (using wrapper module)
# ---------------------------------------------------------------------------
def start_proxy(config_path):
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    env["CONFIG_FILE_PATH"] = config_path
    env["_REPRO_WORKER_MEM_CAP_MB"] = str(WORKER_MEM_CAP_MB)
    env["LITELLM_LOG"] = "ERROR"
    env["LITELLM_MASTER_KEY"] = "sk-test-master-key"

    # Use uvicorn to run our wrapper module (which imports the proxy app + adds leak middleware)
    cmd = [
        sys.executable, "-m", "uvicorn",
        "tests.memory_leak_repro.proxy_wrapper:app",
        "--host", "0.0.0.0",
        "--port", str(PROXY_PORT),
        "--workers", str(NUM_WORKERS),
        "--log-level", "warning",
    ]

    proc = subprocess.Popen(
        cmd, env=env, cwd=WORKSPACE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(f"[{ts()}] Proxy starting (master pid={proc.pid})...")

    if not wait_for_port(PROXY_PORT, timeout=120):
        print(f"[{ts()}] FATAL: proxy did not start on :{PROXY_PORT}")
        try:
            out = proc.stdout.read(8192).decode(errors="replace")
            print(f"--- Proxy output ---\n{out}\n---")
        except Exception:
            pass
        proc.kill()
        sys.exit(1)

    print(f"[{ts()}] ✓ Proxy ready on :{PROXY_PORT} (master pid={proc.pid})")
    return proc


# ---------------------------------------------------------------------------
# 4. Blast requests
# ---------------------------------------------------------------------------
def start_blast():
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "blast.py"),
        "--url", f"http://127.0.0.1:{PROXY_PORT}",
        "--concurrency", str(BLAST_CONCURRENCY),
        "--total", str(BLAST_TOTAL),
    ]
    proc = subprocess.Popen(cmd, env=env, cwd=WORKSPACE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"[{ts()}] ✓ Blast started (pid={proc.pid}, concurrency={BLAST_CONCURRENCY})")
    return proc


# ---------------------------------------------------------------------------
# 5. Stream blast output in background
# ---------------------------------------------------------------------------
def stream_output(proc, prefix):
    def _reader():
        try:
            for line in proc.stdout:
                text = line.decode(errors="replace").rstrip()
                if text:
                    print(f"[{prefix}] {text}")
        except Exception:
            pass
    t = Thread(target=_reader, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# 6. Monitor workers
# ---------------------------------------------------------------------------
def monitor_workers(master_pid, duration):
    known = {}
    deaths = []
    spawns = []
    rss_history = []

    t0 = time.time()
    while time.time() - t0 < duration:
        children = get_child_pids(master_pid)
        current = set(children)
        known_set = set(known.keys())

        for pid in known_set - current:
            last_rss = known.pop(pid)
            deaths.append((ts(), pid, last_rss))
            print(f"[{ts()}] *** WORKER DIED   pid={pid}  last_rss={last_rss:.1f}MB ***")

        for pid in current - known_set:
            rss = get_rss_mb(pid) or 0
            known[pid] = rss
            spawns.append((ts(), pid))
            if time.time() - t0 > 3:  # Skip initial spawn noise
                print(f"[{ts()}] +++ WORKER SPAWN  pid={pid}  rss={rss:.1f}MB +++")

        parts = []
        for pid in sorted(current):
            rss = get_rss_mb(pid)
            if rss is not None:
                known[pid] = rss
                parts.append(f"pid={pid}:{rss:.0f}MB")
                rss_history.append((time.time() - t0, pid, rss))

        if parts:
            print(f"[{ts()}] RSS: {' | '.join(parts)}")

        if len(deaths) >= 2:
            print(f"\n[{ts()}] === CONFIRMED: {len(deaths)} worker deaths ===")
            break

        if not os.path.exists(f"/proc/{master_pid}"):
            print(f"[{ts()}] Master gone")
            break

        time.sleep(POLL_INTERVAL)

    return deaths, spawns, rss_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LiteLLM Proxy Memory Leak Reproduction")
    print("=" * 70)
    print(f"Workers: {NUM_WORKERS}, Mem cap: {WORKER_MEM_CAP_MB}MB, "
          f"Concurrency: {BLAST_CONCURRENCY}, Requests: {BLAST_TOTAL}")
    print()

    start_fake_openai()
    config_path = write_proxy_config()
    proxy_proc = start_proxy(config_path)

    # Give proxy a moment to fully start
    time.sleep(3)

    blast_proc = start_blast()
    stream_output(blast_proc, "blast")

    # Give blast a moment to ramp up
    time.sleep(2)

    try:
        deaths, spawns, rss_history = monitor_workers(proxy_proc.pid, MAX_DURATION)
    except KeyboardInterrupt:
        deaths, spawns, rss_history = [], [], []

    # Summary
    print()
    print("=" * 70)
    print("REPRODUCTION SUMMARY")
    print("=" * 70)
    print(f"Worker deaths: {len(deaths)}")
    for t, pid, rss in deaths:
        print(f"  [{t}] pid={pid} last_rss={rss:.1f}MB")
    print(f"Worker spawns: {len(spawns)}")

    if rss_history:
        by_pid = {}
        for elapsed, pid, rss in rss_history:
            by_pid.setdefault(pid, []).append((elapsed, rss))
        print("\nRSS growth per worker:")
        for pid, points in sorted(by_pid.items()):
            if len(points) >= 2:
                t0_p, rss0 = points[0]
                t1_p, rss1 = points[-1]
                dt = t1_p - t0_p
                if dt > 0:
                    rate = (rss1 - rss0) / dt
                    print(f"  pid={pid}: {rss0:.0f}MB → {rss1:.0f}MB over {dt:.0f}s ({rate:+.1f} MB/s)")

    if len(deaths) >= 1:
        print("\n✓ REPRODUCTION SUCCESSFUL: Workers died from memory growth")
        print("  Root cause: unbounded spend_log_transactions list (simulated via middleware)")
    else:
        print("\n⚠ Workers did not die during test window")
        if rss_history:
            max_rss = max(r for _, _, r in rss_history)
            print(f"  Max RSS observed: {max_rss:.0f}MB (cap={WORKER_MEM_CAP_MB}MB)")

    # Cleanup
    print(f"\n[{ts()}] Cleaning up...")
    for p in [blast_proc, proxy_proc]:
        try:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
