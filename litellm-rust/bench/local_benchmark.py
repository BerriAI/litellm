#!/usr/bin/env python3
"""Simple benchmark for local model testing."""

import requests
import time
import statistics

URL = "http://localhost:4001/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-test-key"
}

def benchmark_request():
    """Send a single request and measure latency."""
    payload = {
        "model": "qwen-0.5b",
        "messages": [{"role": "user", "content": "Write a simple hello world program in Python"}],
        "max_tokens": 50
    }
    
    start = time.perf_counter()
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=30)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    
    return elapsed, response.status_code == 200

def run_benchmark(num_requests=20):
    """Run benchmark with specified number of requests."""
    print(f"Running benchmark with {num_requests} requests...")
    print(f"Endpoint: {URL}")
    print(f"Model: qwen-0.5b (local)")
    print()
    
    latencies = []
    successes = 0
    failures = 0
    
    for i in range(num_requests):
        latency, success = benchmark_request()
        if success:
            successes += 1
            latencies.append(latency)
            print(f"Request {i+1}/{num_requests}: {latency:.1f}ms [OK]")
        else:
            failures += 1
            print(f"Request {i+1}/{num_requests}: FAILED")
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total requests: {num_requests}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Success rate: {successes/num_requests*100:.1f}%")
    
    if latencies:
        print()
        print("Latency Statistics:")
        print(f"  Mean:   {statistics.mean(latencies):.1f}ms")
        print(f"  Median: {statistics.median(latencies):.1f}ms")
        print(f"  P95:    {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  P99:    {sorted(latencies)[int(len(latencies)*0.99)]:.1f}ms")
        print(f"  Min:    {min(latencies):.1f}ms")
        print(f"  Max:    {max(latencies):.1f}ms")
        
        # Calculate throughput
        total_time = sum(latencies) / 1000  # Convert to seconds
        rps = successes / total_time
        print()
        print(f"Throughput: {rps:.2f} RPS")

if __name__ == "__main__":
    run_benchmark(20)
