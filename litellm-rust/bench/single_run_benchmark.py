#!/usr/bin/env python3
"""Single-run benchmark to eliminate variance - tests all three scenarios in one run."""

import requests
import time
import statistics

# Configuration
LLAMA_URL = "http://localhost:8080/v1/chat/completions"
RUST_URL = "http://localhost:4001/v1/chat/completions"
PYTHON_URL = "http://localhost:4002/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
PAYLOAD = {
    "model": "qwen-0.5b",
    "messages": [{"role": "user", "content": "Write a simple hello world program in Python"}],
    "max_tokens": 50
}
NUM_REQUESTS = 20

def run_benchmark(name, url, auth_header=None):
    """Run benchmark for a single endpoint."""
    headers = HEADERS.copy()
    if auth_header:
        headers["Authorization"] = auth_header
    
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {url}")
    print(f"{'='*60}")
    
    # Warmup
    print("Warming up (3 requests)...")
    for _ in range(3):
        requests.post(url, json=PAYLOAD, headers=headers, timeout=30)
    
    # Benchmark
    print(f"Running {NUM_REQUESTS} requests...")
    times = []
    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        r = requests.post(url, json=PAYLOAD, headers=headers, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Request {i+1}/{NUM_REQUESTS}: {elapsed:.1f}ms")
    
    # Results
    print(f"\nResults:")
    print(f"  Mean:   {statistics.mean(times):.1f}ms")
    print(f"  Median: {statistics.median(times):.1f}ms")
    print(f"  P95:    {sorted(times)[int(len(times)*0.95)]:.1f}ms")
    print(f"  P99:    {sorted(times)[int(len(times)*0.99)]:.1f}ms")
    print(f"  Min:    {min(times):.1f}ms")
    print(f"  Max:    {max(times):.1f}ms")
    total_time = sum(times) / 1000
    rps = NUM_REQUESTS / total_time
    print(f"  Throughput: {rps:.2f} RPS")
    
    return {
        "name": name,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": sorted(times)[int(len(times)*0.95)],
        "p99": sorted(times)[int(len(times)*0.99)],
        "min": min(times),
        "max": max(times),
        "rps": rps
    }

def main():
    print("="*60)
    print("SINGLE-RUN BENCHMARK - Eliminating Variance")
    print("="*60)
    print("All three scenarios run in single session")
    print("This eliminates variance from different system conditions")
    
    # Run all three benchmarks
    results = []
    
    # 1. Baseline (direct to llama.cpp)
    results.append(run_benchmark("Baseline (Direct to llama.cpp)", LLAMA_URL))
    
    # 2. Rust Gateway
    results.append(run_benchmark("Rust Gateway", RUST_URL, "Bearer sk-test-key"))
    
    # 3. Python Proxy
    results.append(run_benchmark("Python Proxy", PYTHON_URL, "Bearer sk-test-key"))
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Scenario':<30} {'Mean':>10} {'Median':>10} {'P95':>10} {'RPS':>8}")
    print("-"*60)
    
    baseline = results[0]
    for result in results:
        overhead = result["mean"] - baseline["mean"]
        overhead_pct = (overhead / baseline["mean"]) * 100
        print(f"{result['name']:<30} {result['mean']:>9.1f}ms {result['median']:>9.1f}ms {result['p95']:>9.1f}ms {result['rps']:>7.2f}")
        if result != baseline:
            print(f"{'  Overhead vs baseline':<30} {overhead:>+9.1f}ms ({overhead_pct:>+.1f}%)")
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    rust = results[1]
    python = results[2]
    
    rust_overhead = rust["mean"] - baseline["mean"]
    python_overhead = python["mean"] - baseline["mean"]
    
    print(f"Rust Gateway overhead: {rust_overhead:+.1f}ms ({(rust_overhead/baseline['mean']*100):+.1f}%)")
    print(f"Python Proxy overhead: {python_overhead:+.1f}ms ({(python_overhead/baseline['mean']*100):+.1f}%)")
    
    if rust["mean"] < python["mean"]:
        print(f"\nRust is {python['mean'] - rust['mean']:.1f}ms faster than Python")
        print(f"Rust is {((python['mean'] - rust['mean']) / python['mean'] * 100):.1f}% faster")
    else:
        print(f"\nPython is {rust['mean'] - python['mean']:.1f}ms faster than Rust")
        print(f"Python is {((rust['mean'] - python['mean']) / rust['mean'] * 100):.1f}% faster")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
