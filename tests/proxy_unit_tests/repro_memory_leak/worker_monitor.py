#!/usr/bin/env python3
"""
Monitor LiteLLM proxy worker processes for memory growth and deaths.

Polls /proc every few seconds, tracks RSS of each python worker,
and reports when workers die or new ones spawn.

Usage:
    python worker_monitor.py [--interval SECS] [--parent-pid PID]

If --parent-pid is not given, finds the uvicorn main process automatically.
"""

import argparse
import os
import signal
import sys
import time
from typing import Dict, Optional, Set, Tuple


def read_proc_stat(pid: int) -> Optional[dict]:
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            stat = f.read()
        parts = stat.rsplit(")", 1)
        if len(parts) < 2:
            return None
        comm = stat.split("(", 1)[1].rsplit(")", 1)[0]
        fields = parts[1].split()
        return {
            "pid": pid,
            "comm": comm,
            "ppid": int(fields[1]),
            "rss_pages": int(fields[21]),
        }
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return None


def read_proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            return f.read().replace("\0", " ").strip()
    except (FileNotFoundError, PermissionError):
        return ""


def get_rss_mb(pid: int) -> float:
    stat = read_proc_stat(pid)
    if stat is None:
        return 0.0
    page_size = os.sysconf("SC_PAGE_SIZE")
    return stat["rss_pages"] * page_size / (1024 * 1024)


def find_litellm_workers() -> Dict[int, dict]:
    """Find all python processes that look like litellm workers."""
    workers = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            cmdline = read_proc_cmdline(pid)
            if not cmdline:
                continue
            if "litellm" in cmdline.lower() or "uvicorn" in cmdline.lower():
                stat = read_proc_stat(pid)
                if stat:
                    workers[pid] = {
                        "cmdline": cmdline[:120],
                        "ppid": stat["ppid"],
                        "rss_mb": stat["rss_pages"] * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024),
                        "comm": stat["comm"],
                    }
    except (FileNotFoundError, PermissionError):
        pass
    return workers


def find_query_engines() -> Dict[int, dict]:
    """Find all query-engine processes."""
    engines = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            cmdline = read_proc_cmdline(pid)
            if "query-engine" in cmdline or "prisma" in cmdline.lower():
                stat = read_proc_stat(pid)
                if stat:
                    engines[pid] = {
                        "cmdline": cmdline[:120],
                        "ppid": stat["ppid"],
                        "rss_mb": stat["rss_pages"] * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024),
                    }
    except (FileNotFoundError, PermissionError):
        pass
    return engines


def main():
    parser = argparse.ArgumentParser(description="LiteLLM worker memory monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval in seconds")
    args = parser.parse_args()

    print(f"Worker monitor started (poll every {args.interval}s)")
    print(f"{'Time':>10}  {'PID':>7}  {'PPID':>7}  {'RSS_MB':>8}  {'Delta':>8}  {'Type':>12}  Command")
    print("-" * 100)

    known_workers: Dict[int, float] = {}
    known_engines: Set[int] = set()
    prev_rss: Dict[int, float] = {}
    death_count = 0

    while True:
        ts = time.strftime("%H:%M:%S")

        workers = find_litellm_workers()
        engines = find_query_engines()

        current_pids = set(workers.keys())
        prev_pids = set(known_workers.keys())

        new_pids = current_pids - prev_pids
        dead_pids = prev_pids - current_pids

        for pid in dead_pids:
            death_count += 1
            rss = prev_rss.get(pid, 0)
            print(
                f"[{ts}]  {pid:>7}  {'':>7}  {rss:>7.0f}M  {'':>8}  {'** DIED **':>12}  "
                f"(death #{death_count}, last_rss={rss:.0f}MB)",
                flush=True,
            )

        for pid in new_pids:
            info = workers[pid]
            print(
                f"[{ts}]  {pid:>7}  {info['ppid']:>7}  {info['rss_mb']:>7.1f}M  {'':>8}  {'NEW':>12}  "
                f"{info['cmdline'][:60]}",
                flush=True,
            )

        for pid, info in sorted(workers.items()):
            rss = info["rss_mb"]
            delta = rss - prev_rss.get(pid, rss)
            delta_str = f"{delta:+.1f}M" if abs(delta) > 0.1 else ""

            is_worker = "worker" in info.get("cmdline", "").lower() or info["ppid"] != 1
            proc_type = "worker" if is_worker else "main"

            print(
                f"[{ts}]  {pid:>7}  {info['ppid']:>7}  {rss:>7.1f}M  {delta_str:>8}  {proc_type:>12}",
                flush=True,
            )
            prev_rss[pid] = rss

        engine_pids = set(engines.keys())
        new_engines = engine_pids - known_engines
        dead_engines = known_engines - engine_pids

        for pid in new_engines:
            info = engines[pid]
            orphan = " (ORPHAN ppid=1)" if info["ppid"] == 1 else ""
            print(
                f"[{ts}]  {pid:>7}  {info['ppid']:>7}  {info['rss_mb']:>7.1f}M  {'':>8}  {'engine-NEW':>12}{orphan}",
                flush=True,
            )

        for pid in dead_engines:
            print(
                f"[{ts}]  {pid:>7}  {'':>7}  {'':>8}  {'':>8}  {'engine-DIED':>12}",
                flush=True,
            )

        known_workers = {pid: workers[pid]["rss_mb"] for pid in workers}
        known_engines = engine_pids

        print(flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
