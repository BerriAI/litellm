#!/usr/bin/env python3
"""Burst test Router.parallel_acompletions to catch ordering or error handling regressions."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from litellm import Router

load_dotenv(find_dotenv())

PROMPTS = [
    "Say hello.",
    "List two colors.",
    "Name an animal.",
    "Give a short proverb.",
    "Name a programming language.",
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


try:
    from litellm.router_utils.parallel_acompletion import RouterParallelRequest
except Exception:  # pragma: no cover - version guard
    @dataclass
    class RouterParallelRequest:  # type: ignore
        model: str
        messages: List[Dict[str, Any]]


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


def main() -> None:
    burst = max(1, _env_int(10, "STRESS_BURST", "SCENARIO_BURST_SIZE"))
    rounds = max(1, _env_int(3, "STRESS_ROUNDS", "SCENARIO_BURST_ROUNDS"))

    heavy_requested = burst > 30 or rounds > 8
    if heavy_requested and os.getenv("STRESS_HEAVY") != "1":
        print(
            f"Requested burst={burst} rounds={rounds} exceeds the safe defaults. "
            "Set STRESS_HEAVY=1 to opt into heavier burst testing or lower STRESS_BURST/STRESS_ROUNDS."
        )
        return

    model_alias = (
        os.getenv("SCENARIO_BURST_MODEL")
        or os.getenv("LITELLM_DEFAULT_MODEL")
        or os.getenv("LITELLM_DEFAULT_LARGE_MODEL")
        or "ollama/qwen2.5:7b"
    )
    provider = os.getenv("SCENARIO_BURST_PROVIDER") or infer_provider(model_alias)

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Skipping burst test (OPENAI_API_KEY not set)")
        return

    params: Dict[str, Any] = {"model": model_alias}
    if provider:
        params["custom_llm_provider"] = provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        params["api_key"] = os.environ["OPENAI_API_KEY"]
    if provider == "ollama":
        params.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))

    router = Router(model_list=[{"model_name": "scenario-burst", "litellm_params": params}])

    async def run_round(index: int) -> None:
        requests = [
            RouterParallelRequest(
                model="scenario-burst",
                messages=[{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            )
            for i in range(burst)
        ]
        t0 = time.perf_counter()
        results = await router.parallel_acompletions(
            requests,
            preserve_order=True,
            return_exceptions=True,
        )
        latency = time.perf_counter() - t0
        errors = 0
        for pos, item in enumerate(results):
            err = getattr(item, "error", None)
            if err is None and isinstance(item, dict):
                err = item.get("error")
            if err:
                errors += 1
        print(f"round={index} burst={burst} elapsed_sec={latency:.3f} errors={errors}")

    async def run_all() -> None:
        for idx in range(rounds):
            await run_round(idx)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
