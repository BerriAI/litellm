#!/usr/bin/env python3
"""
Wait for the LiteLLM proxy to be up, then run the proxy-vs-provider benchmark
and save the output to a .txt file.

Usage:
  # Use env vars (LITELLM_PROXY_URL, PROVIDER_URL) and default output file:
  python scripts/run_benchmark_after_proxy.py --rps-control 1000 --requests 10000

  # Specify output file:
  python scripts/run_benchmark_after_proxy.py --output benchmark_results.txt --rps-control 1000 --requests 10000

  # Custom proxy/provider URLs:
  python scripts/run_benchmark_after_proxy.py --proxy-url http://localhost:4000/chat/completions --provider-url http://localhost:8090/chat/completions --rps-control 1000 --requests 10000

  # Skip waiting (run benchmark immediately):
  python scripts/run_benchmark_after_proxy.py --no-wait --rps-control 1000 --requests 10000
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None


def _health_url_from_chat_url(chat_url: str) -> str:
    """e.g. http://localhost:4000/chat/completions -> http://localhost:4000/health"""
    parsed = urlparse(chat_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    return f"{base}/health"


def wait_for_proxy(proxy_url: str, timeout_seconds: float = 300, poll_interval: float = 2.0) -> bool:
    """Return True when proxy health endpoint returns 200, or False on timeout."""
    health_url = _health_url_from_chat_url(proxy_url)
    deadline = time.monotonic() + timeout_seconds
    if httpx is None:
        print("Warning: httpx not installed; skipping proxy health check. Install with: pip install httpx")
        return True
    while time.monotonic() < deadline:
        try:
            r = httpx.get(health_url, timeout=5.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for proxy to be up, run benchmark, save output to a .txt file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output .txt file path (default: benchmark_results_YYYYMMDD_HHMMSS.txt)",
    )
    parser.add_argument(
        "--proxy-url",
        default=os.getenv("LITELLM_PROXY_URL", "http://localhost:4000/chat/completions"),
        help="Proxy chat completions URL (default: LITELLM_PROXY_URL or http://localhost:4000/chat/completions)",
    )
    parser.add_argument(
        "--provider-url",
        default=os.getenv("PROVIDER_URL", "http://localhost:8090/chat/completions"),
        help="Direct provider chat completions URL (default: PROVIDER_URL or http://localhost:8090/chat/completions)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=300,
        help="Seconds to wait for proxy health before giving up (default: 300)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for proxy; run benchmark immediately.",
    )
    # Pass-through: forward unknown args to benchmark script (e.g. --rps-control, --requests)
    args, extra = parser.parse_known_args()

    env = os.environ.copy()
    env["LITELLM_PROXY_URL"] = args.proxy_url
    env["PROVIDER_URL"] = args.provider_url

    if not args.no_wait:
        print(f"Waiting for proxy at {args.proxy_url} (health check)...")
        if not wait_for_proxy(args.proxy_url, timeout_seconds=args.wait_timeout):
            print("Proxy did not become healthy in time.", file=sys.stderr)
            return 1
        print("Proxy is up. Starting benchmark...")

    out_path = args.output
    if out_path is None:
        out_path = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_dir, "benchmark_proxy_vs_provider.py")
    cmd = [sys.executable, benchmark_script] + extra

    print(f"Running: {' '.join(cmd)}")
    print(f"Output file: {os.path.abspath(out_path)}")

    with open(out_path, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Benchmark run at {datetime.now().isoformat()}\n")
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.write(f"# LITELLM_PROXY_URL={args.proxy_url}\n")
        f.write(f"# PROVIDER_URL={args.provider_url}\n")
        f.write("-" * 60 + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        proc.wait()

    if proc.returncode != 0:
        print(f"Benchmark exited with code {proc.returncode}", file=sys.stderr)
    print(f"\nOutput saved to {os.path.abspath(out_path)}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
