#!/usr/bin/env python3
"""Measure Router.acompletion throughput/latency under configurable concurrency."""

from __future__ import annotations

import asyncio
import os
import time
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from dotenv import find_dotenv, load_dotenv
from litellm import Router

load_dotenv(find_dotenv())

PROMPTS = [
    "Give me three synonyms for fast.",
    "Name three countries in Europe.",
    "List two benefits of unit tests.",
]


def infer_provider(alias: str) -> Optional[str]:
    alias = alias or ""
    if alias.startswith("openai/"):
        return "openai"
    if alias.startswith("azure/"):
        return "azure_openai"
    if alias.startswith("gemini/"):
        return "gemini"
    if ":" in alias:
        return "ollama"
    return None


def _env_int(default: int, *names: str) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            pass
    return default


async def main() -> None:
    total = max(1, _env_int(6, "STRESS_TOTAL", "SCENARIO_THRU_TOTAL"))
    concurrency = max(1, _env_int(2, "STRESS_CONCURRENCY", "SCENARIO_THRU_CONCURRENCY"))

    heavy_requested = total > 60 or concurrency > 12
    if heavy_requested and os.getenv("STRESS_HEAVY") != "1":
        print(
            f"Requested total={total} concurrency={concurrency} exceeds the safe defaults. "
            "Set STRESS_HEAVY=1 to opt into heavy throughput runs or lower STRESS_TOTAL/STRESS_CONCURRENCY."
        )
        return

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

    params: Dict[str, Any] = {"model": model_alias}
    if provider:
        params["custom_llm_provider"] = provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        params["api_key"] = os.environ["OPENAI_API_KEY"]
    if provider == "ollama":
        params.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))

    router = Router(model_list=[{"model_name": "scenario-thru", "litellm_params": params}])
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> Tuple[bool, float, str]:
        prompt = PROMPTS[i % len(PROMPTS)]
        start = time.perf_counter()
        try:
            async with sem:
                resp = await router.acompletion(
                    model="scenario-thru",
                    messages=[{"role": "user", "content": prompt}],
                )
            ok = bool(getattr(resp.choices[0].message, "content", ""))
            return ok, time.perf_counter() - start, ""
        except Exception as exc:  # pragma: no cover - live providers
            return False, time.perf_counter() - start, str(exc)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(total)))
    elapsed = time.perf_counter() - t0

    latencies: List[float] = []
    successes = 0
    errors: List[str] = []
    for ok, latency, err in results:
        latencies.append(latency)
        if ok:
            successes += 1
        else:
            errors.append(err)

    def percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = (len(s) - 1) * p
        lower = int(idx)
        upper = min(lower + 1, len(s) - 1)
        if lower == upper:
            return s[lower]
        return s[lower] + (s[upper] - s[lower]) * (idx - lower)

    rps = total / elapsed if elapsed > 0 else 0.0
    print("=== throughput benchmark ===")
    print(f"model_alias={model_alias} provider={provider} total={total} concurrency={concurrency}")
    print(f"elapsed_sec={elapsed:.3f} rps={rps:.2f} successes={successes} errors={len(errors)}")
    if latencies:
        print(
            "latency_sec: "
            f"avg={mean(latencies):.3f} "
            f"p50={percentile(latencies, 0.50):.3f} "
            f"p95={percentile(latencies, 0.95):.3f} "
            f"p99={percentile(latencies, 0.99):.3f}"
        )
    if errors:
        print(f"sample_error={errors[0][:240]}")


if __name__ == "__main__":
    asyncio.run(main())
