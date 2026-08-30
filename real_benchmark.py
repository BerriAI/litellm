#!/usr/bin/env python3
"""Real benchmark measuring actual performance metrics."""

import requests
import time
import statistics
import sys

RUST_URL = "http://127.0.0.1:4030/v1/chat/completions"
PYTHON_URL = "http://127.0.0.1:4031/v1/chat/completions"
MASTER_KEY = "benchmark-key"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {MASTER_KEY}",
}

REQUEST = {
    "model": "minimax/minimax-m3:free",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 5,
}

def benchmark_endpoint(url, name, num_requests=20):
    """Benchmark a single endpoint with real requests."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    latencies = []
    successes = 0
    failures = 0
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        try:
            requests.post(url, json=REQUEST, headers=HEADERS, timeout=10)
        except:
            pass
    
    # Actual benchmark
    print(f"Running {num_requests} requests...")
    for i in range(num_requests):
        start = time.perf_counter()
        try:
            r = requests.post(url, json=REQUEST, headers=HEADERS, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000
            
            if r.status_code == 200:
                successes += 1
                latencies.append(elapsed)
            else:
                failures += 1
                print(f"  Request {i+1}: HTTP {r.status_code}")
        except Exception as e:
            failures += 1
            print(f"  Request {i+1}: Error - {e}")
    
    # Calculate metrics
    if latencies:
        sorted_lat = sorted(latencies)
        p50_idx = int(len(sorted_lat) * 0.50)
        p95_idx = int(len(sorted_lat) * 0.95)
        p99_idx = int(len(sorted_lat) * 0.99)
        
        total_time = sum(latencies) / 1000  # seconds
        rps = successes / total_time if total_time > 0 else 0
        
        print(f"\nResults:")
        print(f"  Success rate: {successes}/{num_requests} ({successes/num_requests*100:.0f}%)")
        print(f"  Mean latency: {statistics.mean(latencies):.1f} ms")
        print(f"  Median (p50): {sorted_lat[p50_idx]:.1f} ms")
        print(f"  P95 latency:  {sorted_lat[p95_idx]:.1f} ms")
        print(f"  P99 latency:  {sorted_lat[p99_idx]:.1f} ms")
        print(f"  Min latency:  {min(latencies):.1f} ms")
        print(f"  Max latency:  {max(latencies):.1f} ms")
        print(f"  Throughput:   {rps:.1f} RPS")
        
        return {
            "success_rate": successes / num_requests,
            "mean": statistics.mean(latencies),
            "p50": sorted_lat[p50_idx],
            "p95": sorted_lat[p95_idx],
            "p99": sorted_lat[p99_idx],
            "min": min(latencies),
            "max": max(latencies),
            "rps": rps,
        }
    else:
        print(f"\nAll requests failed!")
        return None

def main():
    print("\n" + "="*60)
    print("REAL BENCHMARK: Rust Gateway vs Python Proxy")
    print("="*60)
    
    # Check both are running
    try:
        requests.get("http://127.0.0.1:4030/health/liveness", timeout=2)
        print("[OK] Rust gateway is running")
    except:
        print("[FAIL] Rust gateway is NOT running")
        sys.exit(1)
    
    try:
        requests.get("http://127.0.0.1:4031/health/liveness", timeout=2)
        print("[OK] Python proxy is running")
    except:
        print("[FAIL] Python proxy is NOT running")
        sys.exit(1)
    
    # Run benchmarks
    rust_results = benchmark_endpoint(RUST_URL, "Rust Gateway", num_requests=20)
    python_results = benchmark_endpoint(PYTHON_URL, "Python Proxy", num_requests=20)
    
    # Comparison
    if rust_results and python_results:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        
        speedup_mean = python_results["mean"] / rust_results["mean"]
        speedup_p50 = python_results["p50"] / rust_results["p50"]
        speedup_p95 = python_results["p95"] / rust_results["p95"]
        speedup_rps = rust_results["rps"] / python_results["rps"]
        
        print(f"\nLatency (lower is better):")
        print(f"  Mean:  Rust {rust_results['mean']:.1f}ms vs Python {python_results['mean']:.1f}ms")
        print(f"         Speedup: {speedup_mean:.2f}x")
        print(f"  P50:   Rust {rust_results['p50']:.1f}ms vs Python {python_results['p50']:.1f}ms")
        print(f"         Speedup: {speedup_p50:.2f}x")
        print(f"  P95:   Rust {rust_results['p95']:.1f}ms vs Python {python_results['p95']:.1f}ms")
        print(f"         Speedup: {speedup_p95:.2f}x")
        
        print(f"\nThroughput (higher is better):")
        print(f"  Rust:  {rust_results['rps']:.1f} RPS")
        print(f"  Python: {python_results['rps']:.1f} RPS")
        print(f"  Speedup: {speedup_rps:.2f}x")
        
        print(f"\nSuccess Rate:")
        print(f"  Rust:   {rust_results['success_rate']*100:.0f}%")
        print(f"  Python: {python_results['success_rate']*100:.0f}%")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()
