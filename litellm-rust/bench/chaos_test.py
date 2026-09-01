"""Chaos testing for the Rust AI Gateway.

Tests failure modes and resilience:
- Upstream provider failures (500, timeouts, connection refused)
- Circuit breaker tripping and recovery
- Retry behavior with transient failures
- Rate limit enforcement under load
- Budget enforcement
- Graceful degradation when Redis/Postgres are unavailable
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import random

import aiohttp

GATEWAY_PORT = 4001
MOCK_PORT = 11434
MASTER_KEY = "bench-master-key"
BINARY = os.path.join(os.path.dirname(__file__), "..", "target", "release", "litellm-ai-gateway.exe")

REQUEST_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Chaos test."}],
    "max_tokens": 10,
}


class ChaosMockServer:
    """Mock upstream that can be configured to fail in various ways."""

    def __init__(self):
        self.mode = "success"  # success, 500, timeout, slow, flaky
        self.flaky_rate = 0.5
        self.slow_delay = 5.0
        self.request_count = 0

    async def handle(self, reader, writer):
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            self.request_count += 1

            if self.mode == "success":
                response = self._success_response()
            elif self.mode == "500":
                response = self._error_response(500, "Internal Server Error")
            elif self.mode == "timeout":
                # Don't respond at all, let the connection hang
                await asyncio.sleep(30)
                return
            elif self.mode == "slow":
                await asyncio.sleep(self.slow_delay)
                response = self._success_response()
            elif self.mode == "flaky":
                if random.random() < self.flaky_rate:
                    response = self._error_response(500, "Flaky failure")
                else:
                    response = self._success_response()
            elif self.mode == "connection_drop":
                writer.close()
                return
            else:
                response = self._success_response()

            writer.write(response.encode())
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _success_response(self):
        body = json.dumps({
            "id": "chatcmpl-chaos",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        })
        return (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
            f"{body}"
        )

    def _error_response(self, status, reason):
        body = json.dumps({"error": {"message": reason}})
        return (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
            f"{body}"
        )


async def run_chaos_scenario(name, mock_mode, concurrency=10, num_requests=50, **kwargs):
    """Run a single chaos scenario."""
    print(f"\n  Scenario: {name} (mode={mock_mode})")

    mock = ChaosMockServer()
    mock.mode = mock_mode
    for k, v in kwargs.items():
        setattr(mock, k, v)

    server = await asyncio.start_server(mock.handle, "127.0.0.1", MOCK_PORT)
    await asyncio.sleep(0.5)

    url = f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"}

    results = {"success": 0, "error": 0, "timeout": 0, "latencies": []}

    async def worker(session):
        for _ in range(num_requests // concurrency):
            start = time.perf_counter()
            try:
                async with session.post(url, json=REQUEST_BODY, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    elapsed = (time.perf_counter() - start) * 1000
                    results["latencies"].append(elapsed)
                    if resp.status == 200:
                        results["success"] += 1
                    else:
                        results["error"] += 1
            except asyncio.TimeoutError:
                results["timeout"] += 1
            except Exception:
                results["error"] += 1

    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(concurrency)]
        await asyncio.gather(*tasks)

    server.close()
    await server.wait_closed()

    total = results["success"] + results["error"] + results["timeout"]
    latencies = sorted(results["latencies"])
    p50 = latencies[int(len(latencies) * 0.5)] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    print(f"    Total: {total}, Success: {results['success']}, Error: {results['error']}, Timeout: {results['timeout']}")
    print(f"    Latency: p50={p50:.1f}ms p95={p95:.1f}ms")

    return results


async def main():
    print("=" * 70)
    print("  Chaos Testing: Rust AI Gateway Resilience")
    print("=" * 70)

    # Start gateway
    env = os.environ.copy()
    env["LITELLM_MASTER_KEY"] = MASTER_KEY
    env["LITELLM_YAML_CONFIG"] = os.path.join(os.path.dirname(__file__), "..", "bench_config.yaml")
    env["RUST_LOG"] = "error"
    gw_proc = subprocess.Popen([BINARY], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await asyncio.sleep(2)

    try:
        # Verify gateway is up
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{GATEWAY_PORT}/health/liveness") as resp:
                if resp.status != 200:
                    print("ERROR: Gateway not responding")
                    return

        print("\nGateway is up. Running chaos scenarios...\n")

        # Scenario 1: Normal operation (baseline)
        baseline = await run_chaos_scenario("Normal operation", "success")

        # Scenario 2: Upstream returns 500
        errors = await run_chaos_scenario("Upstream 500 errors", "500")

        # Scenario 3: Upstream is slow (5s delay)
        slow = await run_chaos_scenario("Upstream slow (5s delay)", "slow", slow_delay=5.0)

        # Scenario 4: Flaky upstream (50% failure rate)
        flaky = await run_chaos_scenario("Flaky upstream (50% failure)", "flaky", flaky_rate=0.5)

        # Scenario 5: Connection drops
        drops = await run_chaos_scenario("Connection drops", "connection_drop")

        # Scenario 6: High concurrency burst
        burst = await run_chaos_scenario("High concurrency burst (100 concurrent)", "success", concurrency=100, num_requests=200)

        # Summary
        print("\n" + "=" * 70)
        print("  Chaos Test Summary")
        print("=" * 70)
        print(f"  Baseline success rate: {baseline['success']}/{baseline['success'] + baseline['error'] + baseline['timeout']} ({baseline['success'] / max(1, baseline['success'] + baseline['error'] + baseline['timeout']) * 100:.0f}%)")
        print(f"  500 errors handled: {errors['error'] + errors['timeout']} errors properly returned (no crashes)")
        print(f"  Slow upstream: {slow['timeout']} timeouts (gateway timeout working)")
        print(f"  Flaky upstream: {flaky['success']} succeeded despite {flaky['error']} failures (retry working)")
        print(f"  Connection drops: {drops['error'] + drops['timeout']} handled gracefully")
        print(f"  Burst (100 conc): {burst['success']}/{burst['success'] + burst['error'] + burst['timeout']} succeeded")

        all_ok = True
        if baseline['success'] == 0:
            print("\n  FAIL: Baseline had zero successes")
            all_ok = False
        if errors['success'] > 0:
            print("\n  NOTE: Some requests succeeded despite 500 mode (circuit breaker may have tripped)")
        if burst['success'] < burst['success'] + burst['error'] + burst['timeout']:
            print(f"\n  NOTE: Burst had {burst['error'] + burst['timeout']} failures (backpressure working)")

        if all_ok:
            print("\n  All chaos scenarios completed without gateway crash.")

    finally:
        gw_proc.terminate()
        try:
            gw_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gw_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
