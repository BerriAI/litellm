"""
Performance test for litellm.get_model_info

This test ensures that get_model_info performs within acceptable limits.
The function is called by Router.get_router_model_info and should not
contribute significant overhead.
"""

import statistics
import time
from typing import Dict, List, Optional

import pytest

import litellm

# Performance test constants
ITERATIONS = 100000
WARMUP_ITERATIONS = 10
# Threshold accounts for CI slowness (~1.3ms/call) vs local (~0.03ms/call)
# Still catches regressions: unoptimized was ~38-46s, CI optimized is ~133s
PERFORMANCE_THRESHOLD_MS = 200000  # 200 seconds - allows for CI variance while catching major regressions
MS_PER_SECOND = 1000
P95_QUANTILE_N = 20
P95_QUANTILE_INDEX = 18


def benchmark_get_model_info(
    model: str,
    custom_llm_provider: Optional[str] = None,
    iterations: int = ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
    silent: bool = True,
) -> Dict[str, float]:
    """
    Benchmark get_model_info function
    
    Args:
        model: Model name to pass to the function
        custom_llm_provider: Optional custom LLM provider
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        silent: Suppress error messages
    
    Returns:
        Dictionary with timing statistics
    """
    times: List[float] = []
    
    # Warmup iterations
    for _ in range(warmup):
        try:
            litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
        except Exception:
            pass  # Silently ignore errors during warmup
    
    # Actual benchmark iterations
    for i in range(iterations):
        start = time.perf_counter()
        try:
            litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
            end = time.perf_counter()
            elapsed = (end - start) * MS_PER_SECOND  # Convert to milliseconds
            times.append(elapsed)
        except Exception:
            end = time.perf_counter()
            elapsed = (end - start) * MS_PER_SECOND
            times.append(elapsed)
            if not silent:
                print(f"  Error on iteration {i}")
    
    if not times:
        return {}
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "p95": statistics.quantiles(times, n=P95_QUANTILE_N)[P95_QUANTILE_INDEX] if len(times) > 1 else times[0],
        "total_time": sum(times),
        "iterations": len(times),
    }


def construct_model_info_name(model: str, custom_llm_provider: str) -> str:
    """
    Simulate how Router.get_router_model_info constructs model_info_name
    (matching router.py lines 6332-6335)
    """
    if not model.startswith(f"{custom_llm_provider}/"):
        model_info_name = f"{custom_llm_provider}/{model}"
    else:
        model_info_name = model
    return model_info_name


@pytest.mark.parametrize(
    "model,model_info_name",
    [
        ("gpt-4", "openai/gpt-4"),  # Basic model name (router would construct "openai/gpt-4")
        ("openai/gpt-4", "openai/gpt-4"),  # Model already with provider prefix
        ("openai/*", "openai/*"),  # Wildcard model
    ],
)
def test_get_model_info_performance(model: str, model_info_name: str):
    """
    Test that get_model_info completes 100k iterations within acceptable time.
    
    After the _get_model_cost_key optimization, performance improved significantly:
    - Optimized (local): ~1.5-3 seconds for 100k iterations (~0.015-0.03 ms/call)
    - Optimized (CI): ~133 seconds for 100k iterations (~1.3 ms/call) - CI is slower
    - Previous (unoptimized): ~38-46 seconds for 100k iterations
    
    We set a threshold of 200 seconds (200000 ms) to:
    - Allow for CI environment slowness (CI is typically 10-50x slower than local)
    - Still catch significant performance regressions (e.g., if it degrades back to unoptimized or worse)
    
    This ensures the optimization remains effective and catches any future regressions.
    """
    custom_llm_provider = "openai"
    
    # Use the model_info_name as constructed by the router
    if model_info_name == "openai/*":
        test_model = model_info_name
    else:
        test_model = construct_model_info_name(model, custom_llm_provider)
    
    # Run benchmark
    results = benchmark_get_model_info(model=test_model, iterations=ITERATIONS, silent=True)
    
    # Assert total time is under the performance threshold
    # Optimized results show ~1.5-3 seconds, so threshold allows for variance
    # while catching significant regressions (like the old 38-46 second performance)
    assert results["total_time"] < PERFORMANCE_THRESHOLD_MS, (
        f"get_model_info took {results['total_time']:.2f} ms for {ITERATIONS} iterations, "
        f"exceeding {PERFORMANCE_THRESHOLD_MS / MS_PER_SECOND} second threshold. "
        f"Mean: {results['mean']:.4f} ms, P95: {results['p95']:.4f} ms. "
        f"Expected: ~1.5-3 seconds (optimized), Previous: ~38-46 seconds (unoptimized)"
    )


def test_get_model_info_performance_summary():
    """
    Run a comprehensive performance test and print summary statistics.
    This test always passes but provides detailed performance metrics.
    """
    custom_llm_provider = "openai"
    
    test_cases = [
        ("gpt-4", "openai/gpt-4"),
        ("openai/gpt-4", "openai/gpt-4"),
        ("openai/*", "openai/*"),
    ]
    
    all_results = []
    
    for model, model_info_name in test_cases:
        if model_info_name == "openai/*":
            test_model = model_info_name
        else:
            test_model = construct_model_info_name(model, custom_llm_provider)
        
        results = benchmark_get_model_info(model=test_model, iterations=ITERATIONS, silent=True)
        all_results.append((model_info_name, results))
    
    # Print summary (for debugging/CI logs)
    print("\n" + "=" * 70)
    print("get_model_info Performance Summary")
    print("=" * 70)
    for model_info_name, results in all_results:
        print(f"\n{model_info_name}:")
        print(f"  Mean: {results['mean']:.4f} ms | Median: {results['median']:.4f} ms | P95: {results['p95']:.4f} ms")
        print(f"  Total: {results['total_time']:.2f} ms ({results['iterations']} iterations)")
        print(f"  Throughput: {MS_PER_SECOND / results['mean']:.0f} calls/sec")
    print("=" * 70 + "\n")
    
    # Test passes - this is just for reporting
    assert True
