"""
Performance benchmark comparing baseline vs optimized LiteLLM hot paths.

Inspired by the fast-litellm project (neul-labs/fast-litellm), this benchmark
measures the impact of Python-level performance optimizations that were
integrated into LiteLLM, including:

  1. orjson serialization (vs stdlib json)
  2. httpx URL pre-parsing with LRU cache
  3. Router deployment O(1) index lookup (vs O(n) linear scan)
  4. Spend-log sanitization (optimized isinstance ordering)
  5. Prometheus label caching (model_dump avoidance)

Run:
    poetry run python tests/load_tests/benchmark_perf_integration.py
"""

import json
import random
import statistics
import string
import sys
import time
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 1. JSON serialization: stdlib json vs orjson
# ---------------------------------------------------------------------------

try:
    import orjson
    _has_orjson = True
except ImportError:
    _has_orjson = False


def _build_chat_payload(n_messages: int = 5) -> dict:
    """Realistic chat completion request body."""
    messages = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = "".join(random.choices(string.ascii_letters + " ", k=200))
        messages.append({"role": role, "content": content})
    return {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
        "metadata": {"user_id": "bench-user-123", "trace_id": "abc-def-ghi"},
    }


def bench_json_serialization(iterations: int = 10000) -> Dict[str, Any]:
    payload = _build_chat_payload(10)

    # Baseline: stdlib json
    t0 = time.perf_counter()
    for _ in range(iterations):
        json.dumps(payload)
    json_time = time.perf_counter() - t0

    # Optimized: orjson
    if _has_orjson:
        t0 = time.perf_counter()
        for _ in range(iterations):
            orjson.dumps(payload)
        orjson_time = time.perf_counter() - t0
    else:
        orjson_time = json_time

    return {
        "name": "JSON serialization (10-msg payload)",
        "iterations": iterations,
        "baseline_ms": json_time * 1000,
        "optimized_ms": orjson_time * 1000,
        "baseline_ops": iterations / json_time,
        "optimized_ops": iterations / orjson_time,
    }


def bench_json_deserialization(iterations: int = 10000) -> Dict[str, Any]:
    response = json.dumps({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "x" * 500},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
    }).encode()

    t0 = time.perf_counter()
    for _ in range(iterations):
        json.loads(response)
    json_time = time.perf_counter() - t0

    if _has_orjson:
        t0 = time.perf_counter()
        for _ in range(iterations):
            orjson.loads(response)
        orjson_time = time.perf_counter() - t0
    else:
        orjson_time = json_time

    return {
        "name": "JSON deserialization (response body)",
        "iterations": iterations,
        "baseline_ms": json_time * 1000,
        "optimized_ms": orjson_time * 1000,
        "baseline_ops": iterations / json_time,
        "optimized_ops": iterations / orjson_time,
    }


# ---------------------------------------------------------------------------
# 2. httpx URL pre-parsing
# ---------------------------------------------------------------------------

import httpx


def bench_url_parsing(iterations: int = 50000) -> Dict[str, Any]:
    urls = [
        "https://api.openai.com/v1/chat/completions",
        "https://api.anthropic.com/v1/messages",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "https://api.cohere.ai/v1/chat",
        "https://api.mistral.ai/v1/chat/completions",
    ]

    # Baseline: parse every time
    t0 = time.perf_counter()
    for _ in range(iterations):
        for url in urls:
            httpx.URL(url)
    baseline_time = time.perf_counter() - t0

    # Optimized: LRU-cached
    from functools import lru_cache

    @lru_cache(maxsize=64)
    def _cached_parse(url: str) -> httpx.URL:
        return httpx.URL(url)

    # Warm the cache
    for url in urls:
        _cached_parse(url)

    t0 = time.perf_counter()
    for _ in range(iterations):
        for url in urls:
            _cached_parse(url)
    optimized_time = time.perf_counter() - t0

    total_ops = iterations * len(urls)
    return {
        "name": "httpx URL parsing (5 provider URLs)",
        "iterations": total_ops,
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "baseline_ops": total_ops / baseline_time,
        "optimized_ops": total_ops / optimized_time,
    }


# ---------------------------------------------------------------------------
# 3. Router deployment lookup: linear scan vs O(1) index
# ---------------------------------------------------------------------------

def bench_deployment_lookup(iterations: int = 50000) -> Dict[str, Any]:
    n_models = 20
    n_deployments_per_model = 5
    model_list = []
    for i in range(n_models):
        model_name = f"model-group-{i}"
        for j in range(n_deployments_per_model):
            model_list.append({
                "model_name": model_name,
                "litellm_params": {
                    "model": f"provider/model-{i}-{j}",
                    "api_key": f"sk-{'x' * 40}",
                },
                "model_info": {"id": f"id-{i}-{j}"},
            })

    lookup_models = [f"model-group-{random.randint(0, n_models - 1)}" for _ in range(iterations)]

    # Baseline: linear scan (old approach)
    t0 = time.perf_counter()
    for model_name in lookup_models:
        result = [m for m in model_list if m["model_name"] == model_name]
    baseline_time = time.perf_counter() - t0

    # Optimized: O(1) index lookup
    index: Dict[str, List[int]] = {}
    for idx, m in enumerate(model_list):
        name = m["model_name"]
        if name not in index:
            index[name] = []
        index[name].append(idx)

    t0 = time.perf_counter()
    for model_name in lookup_models:
        indices = index.get(model_name, [])
        result = [model_list[i] for i in indices]
    optimized_time = time.perf_counter() - t0

    return {
        "name": "Router deployment lookup (20 groups x 5 deployments)",
        "iterations": iterations,
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "baseline_ops": iterations / baseline_time,
        "optimized_ops": iterations / optimized_time,
    }


# ---------------------------------------------------------------------------
# 4. Spend-log sanitization
# ---------------------------------------------------------------------------

def bench_sanitize_spend_log(iterations: int = 5000) -> Dict[str, Any]:
    large_content = "A" * 50000
    request_body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": large_content},
            {"role": "assistant", "content": "Short reply."},
            {"role": "user", "content": "Follow up with " + "B" * 30000},
        ],
        "metadata": {"trace": "abc", "nested": {"key": "value"}},
    }

    MAX_LEN = 5000
    START_CHARS = int(MAX_LEN * 0.35)
    END_CHARS = min(int(MAX_LEN * 0.65), MAX_LEN - START_CHARS)

    # Baseline: old approach (isinstance dict first, then str)
    def _sanitize_old(value):
        if isinstance(value, dict):
            return {k: _sanitize_old(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_sanitize_old(item) for item in value]
        elif isinstance(value, str):
            if len(value) > MAX_LEN:
                return value[:START_CHARS] + "...[truncated]..." + value[-END_CHARS:]
            return value
        return value

    # Optimized: str-first isinstance ordering (most common leaf type)
    def _sanitize_new(value):
        if isinstance(value, str):
            if len(value) > MAX_LEN:
                return value[:START_CHARS] + "...[truncated]..." + value[-END_CHARS:]
            return value
        elif isinstance(value, dict):
            return {k: _sanitize_new(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_sanitize_new(item) for item in value]
        return value

    t0 = time.perf_counter()
    for _ in range(iterations):
        _sanitize_old(request_body)
    baseline_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _sanitize_new(request_body)
    optimized_time = time.perf_counter() - t0

    return {
        "name": "Spend-log sanitization (3-msg payload with large content)",
        "iterations": iterations,
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "baseline_ops": iterations / baseline_time,
        "optimized_ops": iterations / optimized_time,
    }


# ---------------------------------------------------------------------------
# 5. Prometheus label caching
# ---------------------------------------------------------------------------

def bench_prometheus_labels(iterations: int = 20000) -> Dict[str, Any]:
    """Simulates the Prometheus label_factory overhead.

    In the old code, model_dump() was called up to 37 times per success event
    on the UserAPIKeyLabelValues object. The optimization caches the result
    of model_dump() via get_label_dict().
    """
    from pydantic import BaseModel

    class UserAPIKeyLabelValues(BaseModel):
        end_user: str = ""
        hashed_api_key: str = ""
        api_key_alias: str = ""
        team: str = ""
        team_alias: str = ""
        user: str = ""
        organization: str = ""
        requested_model: str = ""
        model: str = ""
        model_id: str = ""
        api_provider: str = ""

    label_obj = UserAPIKeyLabelValues(
        end_user="user-123",
        hashed_api_key="sk-abc",
        api_key_alias="my-key",
        team="team-1",
        team_alias="prod-team",
        user="admin",
        organization="org-1",
        requested_model="gpt-4o",
        model="gpt-4o-2024-08-06",
        model_id="model-id-123",
        api_provider="openai",
    )

    supported_labels = frozenset({
        "end_user", "hashed_api_key", "api_key_alias", "team",
        "team_alias", "user", "organization",
    })

    CALLS_PER_EVENT = 37

    # Baseline: call model_dump() each time
    t0 = time.perf_counter()
    for _ in range(iterations):
        for _ in range(CALLS_PER_EVENT):
            d = label_obj.model_dump()
            {k: v for k, v in d.items() if k in supported_labels}
    baseline_time = time.perf_counter() - t0

    # Optimized: cache model_dump result, call once per event
    t0 = time.perf_counter()
    for _ in range(iterations):
        cached = label_obj.model_dump()
        filtered = {k: v for k, v in cached.items() if k in supported_labels}
        for _ in range(CALLS_PER_EVENT):
            _ = filtered
    optimized_time = time.perf_counter() - t0

    total_ops = iterations * CALLS_PER_EVENT
    return {
        "name": "Prometheus label factory (37 calls/event)",
        "iterations": total_ops,
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "baseline_ops": total_ops / baseline_time,
        "optimized_ops": total_ops / optimized_time,
    }


# ---------------------------------------------------------------------------
# 6. Simple shuffle routing strategy
# ---------------------------------------------------------------------------

def bench_simple_shuffle(iterations: int = 50000) -> Dict[str, Any]:
    deployments = []
    for i in range(10):
        deployments.append({
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": f"openai/gpt-4o-{i}",
                "api_key": f"sk-{'x' * 40}",
                "rpm": 100 + i * 10,
            },
            "model_info": {"id": f"id-{i}"},
        })

    # Baseline: old approach with logging overhead
    import logging
    logger = logging.getLogger("test_bench")
    logger.setLevel(logging.WARNING)

    t0 = time.perf_counter()
    for _ in range(iterations):
        weights = [m["litellm_params"].get("rpm", 0) for m in deployments]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        logger.debug(f"\nweight {weights}")
        logger.debug(f"\n weights {weights} by rpm")
        selected_index = random.choices(range(len(weights)), weights=weights)[0]
        logger.debug(f"\n selected index, {selected_index}")
        deployment = deployments[selected_index]
    baseline_time = time.perf_counter() - t0

    # Optimized: skip debug formatting entirely when level isn't enabled
    t0 = time.perf_counter()
    for _ in range(iterations):
        weights = [m["litellm_params"].get("rpm", 0) for m in deployments]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        selected_index = random.choices(range(len(weights)), weights=weights)[0]
        deployment = deployments[selected_index]
    optimized_time = time.perf_counter() - t0

    return {
        "name": "Simple shuffle routing (10 weighted deployments)",
        "iterations": iterations,
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "baseline_ops": iterations / baseline_time,
        "optimized_ops": iterations / optimized_time,
    }


# ---------------------------------------------------------------------------
# 7. safe_json_dumps with orjson vs stdlib
# ---------------------------------------------------------------------------

def bench_safe_json_dumps(iterations: int = 5000) -> Dict[str, Any]:
    from pydantic import BaseModel

    class Usage(BaseModel):
        prompt_tokens: int = 50
        completion_tokens: int = 100
        total_tokens: int = 150

    class Choice(BaseModel):
        index: int = 0
        message: dict = {"role": "assistant", "content": "Hello world " * 50}
        finish_reason: str = "stop"

    class ResponseModel(BaseModel):
        id: str = "chatcmpl-abc123"
        choices: list = [Choice().model_dump()]
        usage: Usage = Usage()
        model: str = "gpt-4o"

    data = ResponseModel().model_dump()

    # Baseline: stdlib json
    def safe_dumps_old(d):
        return json.dumps(d, default=str)

    # Optimized: orjson
    if _has_orjson:
        def safe_dumps_new(d):
            return orjson.dumps(d, default=str).decode()
    else:
        safe_dumps_new = safe_dumps_old

    t0 = time.perf_counter()
    for _ in range(iterations):
        safe_dumps_old(data)
    baseline_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        safe_dumps_new(data)
    optimized_time = time.perf_counter() - t0

    return {
        "name": "safe_json_dumps (Pydantic model response)",
        "iterations": iterations,
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "baseline_ops": iterations / baseline_time,
        "optimized_ops": iterations / optimized_time,
    }


# ---------------------------------------------------------------------------
# 8. Header lookup optimization
# ---------------------------------------------------------------------------

def bench_header_lookup(iterations: int = 100000) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234567890",
        "Accept": "application/json",
        "User-Agent": "litellm/1.0",
        "X-Request-ID": "req-abc-123",
        "X-Forwarded-For": "192.168.1.1",
        "X-Stainless-Arch": "x86_64",
        "X-Stainless-OS": "Linux",
        "X-Stainless-Runtime": "CPython",
    }

    target_keys = {"x-stainless-arch", "x-stainless-os"}

    # Baseline: dict comprehension over all headers
    t0 = time.perf_counter()
    for _ in range(iterations):
        result = {k: v for k, v in headers.items() if k.lower() in target_keys}
    baseline_time = time.perf_counter() - t0

    # Optimized: early-exit loop checking only target keys
    t0 = time.perf_counter()
    for _ in range(iterations):
        result = {}
        remaining = len(target_keys)
        for k, v in headers.items():
            kl = k.lower()
            if kl == "x-stainless-arch" or kl == "x-stainless-os":
                result[k] = v
                remaining -= 1
                if remaining == 0:
                    break
    optimized_time = time.perf_counter() - t0

    return {
        "name": "Header lookup (9 headers, 2 targets)",
        "iterations": iterations,
        "baseline_ms": baseline_time * 1000,
        "optimized_ms": optimized_time * 1000,
        "baseline_ops": iterations / baseline_time,
        "optimized_ops": iterations / optimized_time,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct_change(baseline: float, optimized: float) -> float:
    if baseline == 0:
        return 0.0
    return ((optimized - baseline) / baseline) * 100


def _speedup(baseline: float, optimized: float) -> float:
    if optimized == 0:
        return 0.0
    return baseline / optimized


def print_report(results: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 90)
    print("  LITELLM PERFORMANCE BENCHMARK: BASELINE vs OPTIMIZED")
    print("  Inspired by fast-litellm (neul-labs/fast-litellm)")
    print("=" * 90)
    print()
    print(f"{'Benchmark':<52} {'Baseline':>10} {'Optimized':>10} {'Speedup':>9} {'Change':>9}")
    print(f"{'':52} {'(ops/s)':>10} {'(ops/s)':>10} {'':>9} {'':>9}")
    print("-" * 90)

    total_baseline_time = 0
    total_optimized_time = 0

    for r in results:
        name = r["name"]
        if len(name) > 50:
            name = name[:47] + "..."
        b_ops = r["baseline_ops"]
        o_ops = r["optimized_ops"]
        speedup = _speedup(r["baseline_ms"], r["optimized_ms"])
        change = _pct_change(r["baseline_ms"], r["optimized_ms"])

        total_baseline_time += r["baseline_ms"]
        total_optimized_time += r["optimized_ms"]

        indicator = "+" if change < -2 else ("~" if abs(change) <= 2 else "-")
        print(
            f"{name:<52} {b_ops:>10,.0f} {o_ops:>10,.0f} {speedup:>8.1f}x {change:>+8.1f}%  {indicator}"
        )

    print("-" * 90)
    overall_speedup = _speedup(total_baseline_time, total_optimized_time)
    overall_change = _pct_change(total_baseline_time, total_optimized_time)
    print(
        f"{'OVERALL (cumulative wall time)':<52} {'':>10} {'':>10} {overall_speedup:>8.1f}x {overall_change:>+8.1f}%"
    )
    print()
    print(f"  Baseline total:  {total_baseline_time:>10.1f} ms")
    print(f"  Optimized total: {total_optimized_time:>10.1f} ms")
    print(f"  Time saved:      {total_baseline_time - total_optimized_time:>10.1f} ms")
    print()

    if _has_orjson:
        print("  orjson: AVAILABLE (used for optimized serialization)")
    else:
        print("  orjson: NOT AVAILABLE (using stdlib json fallback)")
    print()


def main():
    benchmarks = [
        bench_json_serialization,
        bench_json_deserialization,
        bench_url_parsing,
        bench_deployment_lookup,
        bench_sanitize_spend_log,
        bench_prometheus_labels,
        bench_simple_shuffle,
        bench_safe_json_dumps,
        bench_header_lookup,
    ]

    results = []
    for bench_fn in benchmarks:
        sys.stdout.write(f"  Running {bench_fn.__name__}... ")
        sys.stdout.flush()
        r = bench_fn()
        speedup = _speedup(r["baseline_ms"], r["optimized_ms"])
        print(f"{speedup:.1f}x speedup")
        results.append(r)

    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
