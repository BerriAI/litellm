"""Thorough benchmark for the LiteLLM Rust AI Gateway.

Measures:
- Throughput (req/s) at various concurrency levels
- Latency percentiles (p50, p95, p99, max)
- Error rate
- Health endpoint latency
- Metrics endpoint latency
- Binary size and startup time

Usage:
    python bench/run_benchmark.py
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import statistics
from dataclasses import dataclass, field

import aiohttp

GATEWAY_PORT = 4001
MOCK_PORT = 11434
MASTER_KEY = "bench-master-key"
BINARY = os.path.join(os.path.dirname(__file__), "..", "target", "release", "litellm-ai-gateway.exe")

REQUEST_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, benchmark test."}],
    "max_tokens": 50,
}

CONCURRENCY_LEVELS = [1, 5, 10, 25, 50, 100]
REQUESTS_PER_LEVEL = 500
WARMUP_REQUESTS = 50


@dataclass
class BenchResult:
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    duration_secs: float
    latencies_ms: list = field(default_factory=list)

    @property
    def throughput(self):
        return self.successful / self.duration_secs if self.duration_secs > 0 else 0

    @property
    def p50(self):
        return self._percentile(50)

    @property
    def p95(self):
        return self._percentile(95)

    @property
    def p99(self):
        return self._percentile(99)

    @property
    def max_ms(self):
        return max(self.latencies_ms) if self.latencies_ms else 0

    @property
    def mean_ms(self):
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def error_rate(self):
        return self.failed / self.total_requests * 100 if self.total_requests > 0 else 0

    def _percentile(self, p):
        if not self.latencies_ms:
            return 0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * p / 100)
        idx = min(idx, len(sorted_lat) - 1)
        return sorted_lat[idx]


async def send_request(session, url, headers, body):
    start = time.perf_counter()
    try:
        async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            await resp.read()
            elapsed = (time.perf_counter() - start) * 1000
            return resp.status, elapsed
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        return 0, elapsed


async def run_bench_level(concurrency, num_requests, url, headers, body):
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        latencies = []
        success_count = 0
        fail_count = 0

        async def bounded_request():
            nonlocal success_count, fail_count
            async with sem:
                status, latency = await send_request(session, url, headers, body)
                if status == 200:
                    success_count += 1
                    latencies.append(latency)
                else:
                    fail_count += 1

        start = time.perf_counter()
        tasks = [asyncio.create_task(bounded_request()) for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - start

        return BenchResult(
            concurrency=concurrency,
            total_requests=num_requests,
            successful=success_count,
            failed=fail_count,
            duration_secs=duration,
            latencies_ms=latencies,
        )


async def bench_endpoint(name, url, iterations=100):
    async with aiohttp.ClientSession() as session:
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    await resp.read()
                    latencies.append((time.perf_counter() - start) * 1000)
            except Exception:
                pass
        if latencies:
            latencies.sort()
            p50 = latencies[int(len(latencies) * 0.50)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            mean = statistics.mean(latencies)
            print(f"  {name}: mean={mean:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms")
        else:
            print(f"  {name}: all requests failed")


async def main():
    print("=" * 70)
    print("  LiteLLM Rust AI Gateway - Thorough Benchmark")
    print("=" * 70)

    binary_size = os.path.getsize(BINARY) / (1024 * 1024)
    print(f"\nBinary size: {binary_size:.1f} MB")

    # Start mock upstream
    print("\nStarting mock upstream on port", MOCK_PORT)
    mock_proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "mock_upstream.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    await asyncio.sleep(1)

    # Start gateway
    print("Starting gateway on port", GATEWAY_PORT)
    env = os.environ.copy()
    env["LITELLM_MASTER_KEY"] = MASTER_KEY
    env["LITELLM_YAML_CONFIG"] = os.path.join(os.path.dirname(__file__), "..", "bench_config.yaml")
    env["RUST_LOG"] = "error"

    gw_proc = subprocess.Popen(
        [BINARY],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await asyncio.sleep(2)

    try:
        # Check gateway is up
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{GATEWAY_PORT}/health/liveness") as resp:
                if resp.status != 200:
                    print("ERROR: Gateway not responding on /health/liveness")
                    return

        print("Gateway is up.\n")

        # Startup time
        start = time.perf_counter()
        gw_proc2 = subprocess.Popen([BINARY], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.sleep(0.5)
        for _ in range(20):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{GATEWAY_PORT + 1}/health/liveness") as resp:
                        if resp.status == 200:
                            break
            except Exception:
                pass
            await asyncio.sleep(0.1)
        startup_time = (time.perf_counter() - start) * 1000
        gw_proc2.kill()
        gw_proc2.wait()
        print(f"Startup time: {startup_time:.0f}ms\n")

        # Infrastructure endpoints
        print("Infrastructure endpoint latency (100 requests each):")
        await bench_endpoint("GET /health/liveness", f"http://127.0.0.1:{GATEWAY_PORT}/health/liveness")
        await bench_endpoint("GET /health/readiness", f"http://127.0.0.1:{GATEWAY_PORT}/health/readiness")
        await bench_endpoint("GET /health/deep", f"http://127.0.0.1:{GATEWAY_PORT}/health/deep")
        await bench_endpoint("GET /metrics", f"http://127.0.0.1:{GATEWAY_PORT}/metrics")
        print()

        # Warmup
        print(f"Warming up ({WARMUP_REQUESTS} requests)...")
        url = f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"}
        await run_bench_level(10, WARMUP_REQUESTS, url, headers, REQUEST_BODY)
        print("Warmup complete.\n")

        # Load tests
        print(f"{'Conc':>5} {'Req':>6} {'OK':>5} {'Fail':>5} {'RPS':>8} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'Max':>8} {'Err%':>6}")
        print("-" * 85)

        for conc in CONCURRENCY_LEVELS:
            result = await run_bench_level(conc, REQUESTS_PER_LEVEL, url, headers, REQUEST_BODY)
            print(
                f"{result.concurrency:>5} "
                f"{result.total_requests:>6} "
                f"{result.successful:>5} "
                f"{result.failed:>5} "
                f"{result.throughput:>8.1f} "
                f"{result.mean_ms:>7.2f}ms "
                f"{result.p50:>7.2f}ms "
                f"{result.p95:>7.2f}ms "
                f"{result.p99:>7.2f}ms "
                f"{result.max_ms:>7.2f}ms "
                f"{result.error_rate:>5.1f}%"
            )

        # Input validation overhead
        print("\nInput validation overhead (invalid requests):")
        invalid_body = {"model": "gpt-4o"}
        result = await run_bench_level(10, REQUESTS_PER_LEVEL, url, headers, invalid_body)
        print(f"  {result.total_requests} invalid requests: {result.successful} rejected, {result.duration_secs:.2f}s, {result.total_requests/result.duration_secs:.0f} reject/s")

        # Auth rejection
        print("\nAuth rejection overhead:")
        bad_headers = {"Authorization": "Bearer wrong-key", "Content-Type": "application/json"}
        result = await run_bench_level(10, REQUESTS_PER_LEVEL, url, bad_headers, REQUEST_BODY)
        print(f"  {result.total_requests} bad-auth requests: {result.successful + result.failed} processed, {result.duration_secs:.2f}s, {result.total_requests/result.duration_secs:.0f} reject/s")

        print("\n" + "=" * 70)
        print("  Benchmark complete")
        print("=" * 70)

    finally:
        gw_proc.terminate()
        mock_proc.terminate()
        try:
            gw_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gw_proc.kill()
        try:
            mock_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mock_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
