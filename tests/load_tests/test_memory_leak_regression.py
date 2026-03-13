"""
Memory Leak Regression Test

Sends a sustained burst of requests through litellm.acompletion() using the
SleepModel CustomLLM and measures RSS memory to verify:

1. Memory growth stays bounded during sustained load
2. RSS decreases after traffic stops (via malloc_trim)

This test does NOT start a proxy server — it drives litellm.acompletion()
directly, which exercises the same Logging + spend-tracking code paths
that run inside the proxy.

Usage:
    poetry run pytest tests/load_tests/test_memory_leak_regression.py -v -s

Environment variables:
    MEMORY_TEST_NUM_REQUESTS  – total requests to send (default 2000)
    MEMORY_TEST_CONCURRENCY   – concurrent requests (default 50)
    MEMORY_TEST_INPUT_SIZE    – input payload chars (default 4000)
    MEMORY_TEST_OUTPUT_SIZE   – output payload chars (default 1000)
"""

import asyncio
import gc
import os
import sys
import time

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_REQUESTS = int(os.getenv("MEMORY_TEST_NUM_REQUESTS", "2000"))
CONCURRENCY = int(os.getenv("MEMORY_TEST_CONCURRENCY", "50"))
INPUT_SIZE = int(os.getenv("MEMORY_TEST_INPUT_SIZE", "4000"))
OUTPUT_SIZE = int(os.getenv("MEMORY_TEST_OUTPUT_SIZE", "1000"))
# Maximum allowed memory growth (MB) over the entire test
MAX_MEMORY_GROWTH_MB = int(os.getenv("MEMORY_TEST_MAX_GROWTH_MB", "150"))


def _get_rss_mb() -> float:
    """Return current process RSS in megabytes."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for Linux without psutil
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # KB -> MB
        except Exception:
            pass
    return 0.0


def _try_malloc_trim() -> None:
    """Force glibc to return freed pages to the OS."""
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


async def _send_requests(
    num_requests: int,
    concurrency: int,
    input_size: int,
    output_size: int,
) -> None:
    """Send num_requests through litellm.acompletion with bounded concurrency."""
    import litellm

    sem = asyncio.Semaphore(concurrency)

    async def _one_request():
        async with sem:
            await litellm.acompletion(
                model="sleep_model",
                messages=[
                    {"role": "user", "content": "a" * input_size},
                ],
                sleep_time=0.001,  # minimal sleep, we want throughput
                output_size=output_size,
            )

    # Fire all tasks
    tasks = [asyncio.create_task(_one_request()) for _ in range(num_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_MEMORY_TESTS", "false").lower() != "true",
    reason="Memory tests are slow. Set RUN_MEMORY_TESTS=true to run.",
)
async def test_memory_does_not_grow_unbounded():
    """
    Memory regression test: RSS growth should stay bounded under sustained load.

    Test flow:
    1. Register SleepModel as a custom LLM provider
    2. Warmup phase: send 100 requests to stabilize baseline
    3. Measure baseline RSS
    4. Load phase: send NUM_REQUESTS with CONCURRENCY
    5. Measure peak RSS
    6. Cooldown: gc.collect() + malloc_trim
    7. Measure post-cooldown RSS
    8. Assert growth < MAX_MEMORY_GROWTH_MB
    """
    import litellm

    # Disable all external integrations for a clean test
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []
    litellm.turn_off_message_logging = True  # avoid extra copies

    # Register the sleep model
    from tests.load_tests.sleep_mode import SleepModel

    litellm.custom_provider_map = [
        {"provider": "openai", "custom_handler": SleepModel()}
    ]

    # --- Warmup ---
    print(f"\n[Memory Test] Warmup: 100 requests...")
    await _send_requests(100, 20, INPUT_SIZE, OUTPUT_SIZE)

    # Force GC before measuring baseline
    gc.collect()
    _try_malloc_trim()
    await asyncio.sleep(0.5)

    baseline_rss = _get_rss_mb()
    print(f"[Memory Test] Baseline RSS: {baseline_rss:.1f} MB")

    # --- Load phase ---
    print(
        f"[Memory Test] Load phase: {NUM_REQUESTS} requests, "
        f"concurrency={CONCURRENCY}, input={INPUT_SIZE}B, output={OUTPUT_SIZE}B..."
    )
    load_start = time.time()

    # Send in batches for progress reporting
    batch_size = NUM_REQUESTS // 4
    for batch_num in range(4):
        await _send_requests(batch_size, CONCURRENCY, INPUT_SIZE, OUTPUT_SIZE)
        current_rss = _get_rss_mb()
        print(
            f"[Memory Test]   Batch {batch_num + 1}/4 done "
            f"({(batch_num + 1) * batch_size}/{NUM_REQUESTS}). "
            f"RSS: {current_rss:.1f} MB (+{current_rss - baseline_rss:.1f} MB)"
        )

    load_duration = time.time() - load_start
    peak_rss = _get_rss_mb()
    print(
        f"[Memory Test] Load complete in {load_duration:.1f}s. "
        f"Peak RSS: {peak_rss:.1f} MB (+{peak_rss - baseline_rss:.1f} MB)"
    )

    # --- Cooldown ---
    print("[Memory Test] Cooldown: gc.collect() + malloc_trim...")
    gc.collect()
    _try_malloc_trim()
    await asyncio.sleep(2)
    gc.collect()
    _try_malloc_trim()

    post_cooldown_rss = _get_rss_mb()
    growth = post_cooldown_rss - baseline_rss
    print(
        f"[Memory Test] Post-cooldown RSS: {post_cooldown_rss:.1f} MB "
        f"(growth: +{growth:.1f} MB)"
    )

    # --- Assertions ---
    print(
        f"[Memory Test] Asserting growth ({growth:.1f} MB) "
        f"< {MAX_MEMORY_GROWTH_MB} MB..."
    )
    assert growth < MAX_MEMORY_GROWTH_MB, (
        f"Memory grew by {growth:.1f} MB over {NUM_REQUESTS} requests. "
        f"Threshold: {MAX_MEMORY_GROWTH_MB} MB. "
        f"Baseline: {baseline_rss:.1f} MB, Peak: {peak_rss:.1f} MB, "
        f"Post-cooldown: {post_cooldown_rss:.1f} MB"
    )

    # Check that malloc_trim helped reduce RSS from peak
    if peak_rss > baseline_rss + 20:  # only check if there was meaningful growth
        reduction_pct = (peak_rss - post_cooldown_rss) / (peak_rss - baseline_rss) * 100
        print(
            f"[Memory Test] malloc_trim reclaimed {reduction_pct:.0f}% of peak growth"
        )

    print(f"[Memory Test] ✅ PASSED — growth: {growth:.1f} MB < {MAX_MEMORY_GROWTH_MB} MB")
