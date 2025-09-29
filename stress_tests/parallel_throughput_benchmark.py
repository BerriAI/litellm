#!/usr/bin/env python3
"""
Stress test Router.acompletion throughput/latency with configurable concurrency and request volume.
Reports success/error counts, RPS, and latency percentiles (p50/p95/p99).
Env:
  SCENARIO_THRU_MODEL          - model alias (default: ollama/qwen2.5:7b)
  SCENARIO_THRU_PROVIDER       - provider override; inferred if not set
  SCENARIO_THRU_TOTAL          - total number of requests (default: 100)
  SCENARIO_THRU_CONCURRENCY    - concurrent tasks (default: 10)
  OLLAMA_API_BASE              - default http://127.0.0.1:11434 for ollama
  OPENAI_API_KEY               - required if provider=openai
"""
import asyncio
import os
import time
from statistics import mean
from typing import List, Optional, Tuple

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router


def infer_provider(alias: str) -> Optional[str]:
    if alias.startswith("openai/"):
        return "openai"
    if alias.startswith("azure/"):
        return "azure_openai"
    if alias.startswith("gemini/"):
        return "gemini"
    if ":" in alias:
        return "ollama"
    return None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _build_router(model_alias: str, provider: Optional[str]) -> Router:
    params = {"model": model_alias}
    if provider:
        params["custom_llm_provider"] = provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        params["api_key"] = os.environ["OPENAI_API_KEY"]
    if provider == "ollama":
        params.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))
    return Router(model_list=[{"model_name": "scenario-thru", "litellm_params": params}])


async def main() -> None:
    total = int(os.getenv("SCENARIO_THRU_TOTAL", "100"))
    concurrency = int(os.getenv("SCENARIO_THRU_CONCURRENCY", "10"))

    model_alias = (
        os.getenv("SCENARIO_THRU_MODEL")
        or os.getenv("LITELLM_DEFAULT_MODEL")
        or os.getenv("LITELLM_DEFAULT_LARGE_MODEL")
        or "ollama/qwen2.5:7b"
    )
    provider = os.getenv("SCENARIO_THRU_PROVIDER") or infer_provider(model_alias)

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Skipping throughput benchmark (OPENAI_API_KEY not set)")
        return

    router = _build_router(model_alias, provider)

    prompts = [
        "Give me three synonyms for fast.",
        "Name three countries in Europe.",
        "List two benefits of unit tests.",
    ]

    sem = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    successes = 0
    errors: List[str] = []

    async def one(i: int) -> Tuple[bool, float, str]:
        msg = prompts[i % len(prompts)]
        t0 = time.perf_counter()
        try:
            async with sem:
                resp = await router.acompletion(
                    model="scenario-thru",
                    messages=[{"role": "user", "content": msg}],
                )
            content = getattr(resp.choices[0].message, "content", "")
            ok = bool(content)
            dt = time.perf_counter() - t0
            return ok, dt, ""
        except Exception as e:  # noqa: BLE001
            dt = time.perf_counter() - t0
            return False, dt, str(e)

    t_start = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(total)))
    t_end = time.perf_counter()

    for ok, dt, err in results:
        latencies.append(dt)
        if ok:
            successes += 1
        else:
            errors.append(err)

    elapsed = t_end - t_start
    rps = total / elapsed if elapsed > 0 else 0.0

    print("=== throughput benchmark ===")
    print(
        f"model_alias={model_alias} provider={provider} total={total} "
        f"concurrency={concurrency}"
    )
    print(
        f"elapsed_sec={elapsed:.3f} rps={rps:.2f} successes={successes} "
        f"errors={len(errors)}"
    )
    if latencies:
        print(
            "latency_sec: "
            f"avg={mean(latencies):.3f} "
            f"p50={percentile(latencies, 0.50):.3f} "
            f"p95={percentile(latencies, 0.95):.3f} "
            f"p99={percentile(latencies, 0.99):.3f}"
        )
    if errors:
        print("sample_error:", errors[0][:300])


if __name__ == "__main__":
    asyncio.run(main())
