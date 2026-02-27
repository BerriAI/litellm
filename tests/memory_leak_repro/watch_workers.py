"""
Worker RSS monitor — watches litellm/uvicorn worker processes for memory growth and deaths.

Scans /proc every 2 seconds, prints RSS for each worker, detects deaths and spawns.
Run: python watch_workers.py [--parent-pid PID]
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime


def get_children_pids(parent_pid):
    """Get all child PIDs of a given parent PID by scanning /proc."""
    children = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat = f.read()
                # Field 4 (0-indexed 3) is the parent PID
                parts = stat.split(")")  # Split after comm field which may contain spaces
                if len(parts) >= 2:
                    fields = parts[-1].split()
                    ppid = int(fields[1])  # Field index 1 after closing paren = ppid
                    if ppid == parent_pid:
                        children.append(pid)
            except (FileNotFoundError, PermissionError, ValueError, IndexError):
                continue
    except Exception:
        pass
    return children


def get_rss_mb(pid):
    """Get RSS in MB for a given PID from /proc/[pid]/status."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS:    12345 kB
                    parts = line.split()
                    return int(parts[1]) / 1024.0  # kB → MB
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def get_cmdline(pid):
    """Get command line for a PID."""
    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            return f.read().replace("\0", " ").strip()[:80]
    except (FileNotFoundError, PermissionError):
        return ""


def find_proxy_master_pid():
    """Find the litellm/uvicorn master process PID."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        cmdline = get_cmdline(pid)
        if "litellm" in cmdline and ("--port" in cmdline or "--config" in cmdline):
            return pid
        if "uvicorn" in cmdline and "litellm" in cmdline:
            return pid
    return None


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def main(parent_pid=None):
    if parent_pid is None:
        print("[watch] Searching for litellm/uvicorn master process...")
        for _ in range(60):
            parent_pid = find_proxy_master_pid()
            if parent_pid:
                break
            time.sleep(1)
        if parent_pid is None:
            print("[watch] ERROR: Could not find litellm/uvicorn master process")
            sys.exit(1)

    print(f"[watch] Monitoring children of master PID {parent_pid} (RSS in MB)")
    print(f"[watch] Master cmdline: {get_cmdline(parent_pid)}")
    print()

    known_pids = {}  # pid -> last_rss
    deaths = []
    spawns = []

    try:
        while True:
            children = get_children_pids(parent_pid)
            current_set = set(children)
            known_set = set(known_pids.keys())

            # Detect deaths
            for pid in known_set - current_set:
                last_rss = known_pids.pop(pid)
                msg = f"[{now_str()}] *** WORKER DIED  pid={pid}  last_rss={last_rss:.1f}MB ***"
                print(msg)
                deaths.append((now_str(), pid, last_rss))

            # Detect spawns
            for pid in current_set - known_set:
                rss = get_rss_mb(pid) or 0.0
                known_pids[pid] = rss
                msg = f"[{now_str()}] +++ WORKER SPAWNED pid={pid}  rss={rss:.1f}MB +++"
                print(msg)
                spawns.append((now_str(), pid))

            # Report RSS for all workers
            line_parts = []
            for pid in sorted(current_set):
                rss = get_rss_mb(pid)
                if rss is not None:
                    known_pids[pid] = rss
                    line_parts.append(f"pid={pid} rss={rss:.1f}MB")
                else:
                    line_parts.append(f"pid={pid} rss=???")

            if line_parts:
                print(f"[{now_str()}] {' | '.join(line_parts)}")

            # Check if master is still alive
            if not os.path.exists(f"/proc/{parent_pid}"):
                print(f"[{now_str()}] Master PID {parent_pid} is gone. Exiting.")
                break

            time.sleep(2)

    except KeyboardInterrupt:
        pass

    print(f"\n=== SUMMARY ===")
    print(f"Worker deaths: {len(deaths)}")
    for ts, pid, rss in deaths:
        print(f"  [{ts}] pid={pid} last_rss={rss:.1f}MB")
    print(f"Worker spawns: {len(spawns)}")
    for ts, pid in spawns:
        print(f"  [{ts}] pid={pid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args()
    main(args.parent_pid)
