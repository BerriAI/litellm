#!/usr/bin/env python3
"""
High-concurrency load toward a codex-agent endpoint to surface rate limits and validate resilience.
Skips if codex-agent is not enabled/configured.
Env:
  LITELLM_ENABLE_CODEX_AGENT=1
  CODEX_AGENT_API_BASE=https://your-codex-agent-base
  CODEX_AGENT_API_KEY=...  (optional, if needed)
  SCENARIO_CODEX_MODEL=codex-agent/gpt-4o-mini (default)
  SCENARIO_CODEX_CONCURRENCY=20
  SCENARIO_CODEX_TOTAL=200
"""
import asyncio
import os
import time
from statistics import mean
from typing import Optional, Tuple

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router


def infer_provider(alias: str) -> Optional[str]:
    if alias.startswith("codex-agent/"):
        return "openai"
    if alias.startswith("openai/"):
        return "openai"
    return None


def build_router(model_alias: str, provider: Optional[str]) -> Router:
    api_base = os.getenv("CODEX_AGENT_API_BASE")
    if not api_base:
        raise RuntimeError("CODEX_AGENT_API_BASE not set")

    params = {"model": model_alias, "custom_llm_provider": provider, "api_base": api_base}
    if os.getenv("CODEX_AGENT_API_KEY"):
        params["api_key"] = os.environ["CODEX_AGENT_API_KEY"]
    return Router(model_list=[{"model_name": "scenario-codex", "litellm_params": params}])


async def main() -> None:
    if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
        print("Skipping codex-agent rate limit test (LITELLM_ENABLE_CODEX_AGENT != 1)")
        return

    model_alias = os.getenv("SCENARIO_CODEX_MODEL", "codex-agent/gpt-4o-mini")
    provider = infer_provider(model_alias)

    try:
        router = build_router(model_alias, provider)
    except RuntimeError as exc:
        print(f"Skipping codex-agent rate limit test ({exc})")
        return

    total = int(os.getenv("SCENARIO_CODEX_TOTAL", "200"))
    concurrency = int(os.getenv("SCENARIO_CODEX_CONCURRENCY", "20"))
    sem = asyncio.Semaphore(concurrency)

    prompts = [
        "Write a one-line function in Python that returns the square of a number.",
        "Provide a short docstring for a function that sorts a list in-place.",
        "Explain what a Python list comprehension is in one sentence.",
    ]

    successes = 0
    errors = 0
    lat = []

    async def one(i: int) -> Tuple[bool, float]:
        t0 = time.perf_counter()
        try:
            async with sem:
                resp = await router.acompletion(
                    model="scenario-codex",
                    messages=[{"role": "user", "content": prompts[i % len(prompts)]}],
                )
            content = getattr(resp.choices[0].message, "content", "")
            return bool(content), time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            if i < 5:
                print(f"sample_error[{i}]: {str(e)[:300]}")
            return False, time.perf_counter() - t0

    t_start = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(total)))
    t_end = time.perf_counter()

    for ok, dt in results:
        lat.append(dt)
        if ok:
            successes += 1
        else:
            errors += 1

    elapsed = t_end - t_start
    rps = total / elapsed if elapsed > 0 else 0.0
    print("=== codex-agent rate limit/backoff ===")
    print(
        f"model_alias={model_alias} total={total} concurrency={concurrency} "
        f"elapsed_sec={elapsed:.3f} rps={rps:.2f} successes={successes} errors={errors}"
    )
    if lat:
        print(f"lat_avg={mean(lat):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
