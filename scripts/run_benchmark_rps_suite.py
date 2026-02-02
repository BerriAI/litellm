#!/usr/bin/env python3
"""
Run the proxy-vs-provider benchmark 30 times: 1k, 2k, ..., 30k RPS, one at a time.
Each run writes to a separate results file (e.g. benchmark_results_1000rps.txt, ...).

Usage:
  # Wait for proxy, then run 30 tests (1k–30k RPS), 1M requests each:
  python scripts/run_benchmark_rps_suite.py

  # Skip proxy wait (proxy already up):
  python scripts/run_benchmark_rps_suite.py --no-wait

  # Custom output directory and request count:
  python scripts/run_benchmark_rps_suite.py --output-dir ./results --requests 5000

  # Custom proxy/provider URLs:
  python scripts/run_benchmark_rps_suite.py --proxy-url http://localhost:4000/chat/completions --provider-url http://localhost:8090/chat/completions
"""

import argparse
import os
import subprocess
import sys

# RPS values for the runs: 1k, 2k, ..., 30k
RPS_LEVELS = list(range(1000, 31_000, 1000))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark 30 times at 1k–30k RPS, one at a time, each to a separate file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for result files (default: current directory). Files: benchmark_results_1000rps.txt, ...",
    )
    parser.add_argument(
        "--proxy-url",
        default=os.getenv("LITELLM_PROXY_URL", "http://localhost:4000/chat/completions"),
        help="Proxy chat completions URL",
    )
    parser.add_argument(
        "--provider-url",
        default=os.getenv("PROVIDER_URL", "http://localhost:8090/chat/completions"),
        help="Direct provider chat completions URL",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=1_000_000,
        help="Number of requests per run (default: 1M)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for proxy; assume it is already up.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runner_script = os.path.join(script_dir, "run_benchmark_after_proxy.py")
    if not os.path.isfile(runner_script):
        print(f"Error: {runner_script} not found", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["LITELLM_PROXY_URL"] = args.proxy_url
    env["PROVIDER_URL"] = args.provider_url

    num_runs = len(RPS_LEVELS)
    print(f"RPS suite: {num_runs} runs at {RPS_LEVELS[0]}–{RPS_LEVELS[-1]} RPS, {args.requests} requests each")
    print(f"Output dir: {os.path.abspath(output_dir)}")
    print()

    for i, rps in enumerate(RPS_LEVELS, 1):
        out_file = os.path.join(output_dir, f"benchmark_results_{rps}rps.txt")
        cmd = [
            sys.executable,
            runner_script,
            "--output",
            out_file,
            "--proxy-url",
            args.proxy_url,
            "--provider-url",
            args.provider_url,
            "--rps-control",
            str(rps),
            "--requests",
            str(args.requests),
        ]
        # Only wait for proxy on first run (unless --no-wait)
        if i > 1 or args.no_wait:
            cmd.append("--no-wait")
        print(f"[{i}/{num_runs}] RPS={rps} -> {os.path.basename(out_file)}")
        cwd = os.path.dirname(script_dir)  # repo root so benchmark script finds benchmark_proxy_vs_provider.py
        ret = subprocess.run(cmd, env=env, cwd=cwd)
        if ret.returncode != 0:
            print(f"Run failed with exit code {ret.returncode}", file=sys.stderr)
            return ret.returncode
        print()

    print(f"All {num_runs} runs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
