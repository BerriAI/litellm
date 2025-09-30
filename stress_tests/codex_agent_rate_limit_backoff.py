#!/usr/bin/env python3
"""Hammer codex-agent to surface 429s and verify resiliency under load."""

from __future__ import annotations

import asyncio
import os
import time
from statistics import mean
from typing import Optional, Tuple

from dotenv import find_dotenv, load_dotenv
from litellm import Router

load_dotenv(find_dotenv())


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

PROMPTS = [
    "Write a one-line Python function that returns the square of a number.",
    "Provide a short docstring for a function that sorts a list in-place.",
    "Explain what a Python list comprehension is in one sentence.",
]


def infer_provider(alias: str) -> Optional[str]:
    if alias.startswith("codex-agent/"):
        return "openai"
    if alias.startswith("openai/"):
        return "openai"
    if alias.startswith("azure/"):
        return "azure_openai"
    return None


async def main() -> None:
    if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
        print("Skipping codex-agent rate limit test (LITELLM_ENABLE_CODEX_AGENT != 1)")
        return

    api_base = os.getenv("CODEX_AGENT_API_BASE")
    if not api_base:
        print("Skipping codex-agent rate limit test (CODEX_AGENT_API_BASE not set)")
        return

    model_alias = os.getenv("SCENARIO_CODEX_MODEL", "codex-agent/gpt-4o-mini")
    provider = infer_provider(model_alias)

    total = max(1, _env_int(20, "STRESS_TOTAL", "SCENARIO_CODEX_TOTAL"))
    concurrency = max(1, _env_int(2, "STRESS_CONCURRENCY", "SCENARIO_CODEX_CONCURRENCY"))
    if (total > 60 or concurrency > 8) and os.getenv("STRESS_HEAVY") != "1":
        print(
            f"Requested total={total} concurrency={concurrency} exceeds safe defaults. "
            "Set STRESS_HEAVY=1 to opt into heavier codex-agent load testing or lower STRESS_TOTAL/STRESS_CONCURRENCY."
        )
        return
    sem = asyncio.Semaphore(concurrency)

    params = {
        "model": model_alias,
        "custom_llm_provider": provider or "openai",
        "api_base": api_base,
    }
    if os.getenv("CODEX_AGENT_API_KEY"):
        params["api_key"] = os.environ["CODEX_AGENT_API_KEY"]

    router = Router(model_list=[{"model_name": "scenario-codex", "litellm_params": params}])

    async def one(i: int) -> Tuple[bool, float]:
        prompt = PROMPTS[i % len(PROMPTS)]
        start = time.perf_counter()
        try:
            async with sem:
                resp = await router.acompletion(
                    model="scenario-codex",
                    messages=[{"role": "user", "content": prompt}],
                )
            ok = bool(getattr(resp.choices[0].message, "content", ""))
            return ok, time.perf_counter() - start
        except Exception as exc:  # pragma: no cover - depends on provider
            if i < 5:
                print(f"sample_error[{i}]={str(exc)[:300]}")
            return False, time.perf_counter() - start

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(total)))
    elapsed = time.perf_counter() - t0

    successes = sum(1 for ok, _ in results if ok)
    errors = total - successes
    latencies = [dt for _, dt in results]

    print("=== codex-agent rate limit/backoff ===")
    print(f"model_alias={model_alias} total={total} concurrency={concurrency}")
    print(f"elapsed_sec={elapsed:.3f} rps={(total / elapsed) if elapsed else 0.0:.2f} successes={successes} errors={errors}")
    if latencies:
        print(f"lat_avg={mean(latencies):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
