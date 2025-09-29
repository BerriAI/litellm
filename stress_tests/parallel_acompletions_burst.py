#!/usr/bin/env python3
"""
Burst test using Router.parallel_acompletions to validate ordering and error handling.
Env:
  SCENARIO_BURST_MODEL      - model alias (default: ollama/qwen2.5:7b)
  SCENARIO_BURST_PROVIDER   - provider override; inferred if not set
  SCENARIO_BURST_SIZE       - requests per burst round (default: 20)
  SCENARIO_BURST_ROUNDS     - number of rounds (default: 5)
"""
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router

try:
    from litellm.router_utils.parallel_acompletion import RouterParallelRequest
except Exception:  # noqa: BLE001
    @dataclass
    class RouterParallelRequest:  # type: ignore
        model: str
        messages: List[Dict[str, Any]]


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


def build_router(model_alias: str, provider: Optional[str]) -> Router:
    params = {"model": model_alias}
    if provider:
        params["custom_llm_provider"] = provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        params["api_key"] = os.environ["OPENAI_API_KEY"]
    if provider == "ollama":
        params.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))
    return Router(model_list=[{"model_name": "scenario-burst", "litellm_params": params}])


def main() -> None:
    burst = int(os.getenv("SCENARIO_BURST_SIZE", "20"))
    rounds = int(os.getenv("SCENARIO_BURST_ROUNDS", "5"))

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

    router = build_router(model_alias, provider)

    prompts = [
        "Say hello.",
        "List two colors.",
        "Name an animal.",
        "Give a short proverb.",
        "Name a programming language.",
    ]

    async def run_round(r: int):
        reqs = [
            RouterParallelRequest(
                model="scenario-burst",
                messages=[{"role": "user", "content": prompts[i % len(prompts)]}],
            )
            for i in range(burst)
        ]
        t0 = time.perf_counter()
        results = await router.parallel_acompletions(
            reqs,
            preserve_order=True,
            return_exceptions=True,
        )
        dt = time.perf_counter() - t0
        errs = 0
        for item in results:
            err = getattr(item, "error", None)
            if err is None and isinstance(item, dict):
                err = item.get("error")
            if err:
                errs += 1
        print(f"round={r} burst={burst} elapsed_sec={dt:.3f} errors={errs}")

    async def run_all():
        for r in range(rounds):
            await run_round(r)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
