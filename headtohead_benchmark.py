#!/usr/bin/env python3
"""Head-to-head Rust vs Python benchmark with real OpenRouter API."""

import subprocess
import time
import requests
import statistics

RUST_PORT = 4030
PYTHON_PORT = 4031
MASTER_KEY = "benchmark-key"
API_URL_RUST = f"http://127.0.0.1:{RUST_PORT}/v1/chat/completions"
API_URL_PYTHON = f"http://127.0.0.1:{PYTHON_PORT}/v1/chat/completions"

REQUEST = {
    "model": "minimax/minimax-m3:free",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 5,
}

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {MASTER_KEY}",
}

NUM_REQUESTS = 10


def start_rust():
    env = {**__import__('os').environ, "LITELLM_YAML_CONFIG": "headtohead_config.yaml", "PORT": str(RUST_PORT)}
    return subprocess.Popen(
        ["litellm-rust/target/release/litellm-ai-gateway.exe"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="C:/Users/ian/Documents/LiteLLM"
    )


def start_python():
    env = {**__import__('os').environ, "LITELLM_CONFIG": "headtohead_config.yaml"}
    return subprocess.Popen(
        ["python", "-m", "litellm", "--config", "headtohead_config.yaml", "--port", str(PYTHON_PORT)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd="C:/Users/ian/Documents/LiteLLM"
    )


def wait_for_port(port, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health/liveness", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def benchmark(url, name, num_requests=NUM_REQUESTS):
    print(f"\n{name}:")
    latencies = []
    successes = 0
    failures = 0

    for i in range(num_requests):
        start = time.perf_counter()
        try:
            r = requests.post(url, json=REQUEST, headers=HEADERS, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000
            if r.status_code == 200:
                successes += 1
                latencies.append(elapsed)
                if i == 0:
                    data = r.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "N/A")
                    print(f"  First response: '{content}'")
            else:
                failures += 1
                print(f"  Request {i+1}: HTTP {r.status_code}")
        except Exception as e:
            failures += 1
            print(f"  Request {i+1}: Error - {e}")

    if latencies:
        print(f"  Success: {successes}/{num_requests}")
        print(f"  Failures: {failures}/{num_requests}")
        print(f"  Mean latency: {statistics.mean(latencies):.1f}ms")
        print(f"  Median latency: {statistics.median(latencies):.1f}ms")
        print(f"  P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
        print(f"  Min latency: {min(latencies):.1f}ms")
        print(f"  Max latency: {max(latencies):.1f}ms")
        throughput = successes / (sum(latencies) / 1000) if latencies else 0
        print(f"  Throughput: {throughput:.1f} RPS")
    else:
        print(f"  All requests failed!")

    return latencies, successes


def main():
    print("=" * 60)
    print("  Rust Gateway vs Python Proxy - Head-to-Head Benchmark")
    print("=" * 60)

    # Start Rust gateway
    print("\nStarting Rust gateway...")
    rust_proc = start_rust()
    if not wait_for_port(RUST_PORT):
        print("ERROR: Rust gateway failed to start")
        rust_proc.kill()
        return
    print("  Rust gateway ready!")

    # Start Python proxy
    print("\nStarting Python proxy...")
    python_proc = start_python()
    if not wait_for_port(PYTHON_PORT):
        print("ERROR: Python proxy failed to start")
        rust_proc.kill()
        python_proc.kill()
        return
    print("  Python proxy ready!")

    # Warmup
    print("\nWarming up...")
    requests.post(API_URL_RUST, json=REQUEST, headers=HEADERS, timeout=10)
    requests.post(API_URL_PYTHON, json=REQUEST, headers=HEADERS, timeout=10)
    time.sleep(1)

    # Benchmark Rust
    rust_latencies, rust_success = benchmark(API_URL_RUST, "Rust Gateway")

    # Benchmark Python
    python_latencies, python_success = benchmark(API_URL_PYTHON, "Python Proxy")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if rust_latencies and python_latencies:
        rust_mean = statistics.mean(rust_latencies)
        python_mean = statistics.mean(python_latencies)
        speedup = python_mean / rust_mean if rust_mean > 0 else 0

        print(f"\n  Rust mean latency:  {rust_mean:.1f}ms")
        print(f"  Python mean latency: {python_mean:.1f}ms")
        print(f"  Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"\n  Rust success rate:  {rust_success}/{NUM_REQUESTS} ({rust_success/NUM_REQUESTS*100:.0f}%)")
        print(f"  Python success rate: {python_success}/{NUM_REQUESTS} ({python_success/NUM_REQUESTS*100:.0f}%)")

        if rust_mean < python_mean:
            print(f"\n  Winner: Rust Gateway ({speedup:.1f}x faster)")
        elif python_mean < rust_mean:
            print(f"\n  Winner: Python Proxy ({1/speedup:.1f}x faster)")
        else:
            print(f"\n  Tie!")
    else:
        print("\n  Could not compare - one or both systems failed")

    # Cleanup
    rust_proc.kill()
    python_proc.kill()
    print("\nDone!")


if __name__ == "__main__":
    main()
