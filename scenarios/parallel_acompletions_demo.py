#!/usr/bin/env python3
"""Demonstrate Router.parallel_acompletions with increasing prompt complexity."""

import asyncio
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router
from litellm.router_utils.parallel_acompletion import RouterParallelRequest

PROMPTS = {
    "simple": "List three programming languages",
    "medium": "Provide a fun fact about pandas and mention their habitat.",
    "complex": "Summarize the benefits and trade-offs of unit testing versus integration testing in a paragraph.",
}


def infer_provider(alias: str) -> str | None:
    if alias.startswith("openai/"):
        return "openai"
    if alias.startswith("azure/"):
        return "azure_openai"
    if alias.startswith("gemini/"):
        return "gemini"
    if ":" in alias:
        return "ollama"
    return None


def main() -> None:
    model_alias = (
        os.getenv("SCENARIO_PARALLEL_MODEL")
        or os.getenv("LITELLM_DEFAULT_MODEL")
        or os.getenv("LITELLM_DEFAULT_LARGE_MODEL")
        or "glm4:latest"
    )

    provider = os.getenv("SCENARIO_PARALLEL_PROVIDER") or infer_provider(model_alias)

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Skipping parallel-acompletions scenario (OPENAI_API_KEY not set)")
        return

    params = {"model": model_alias}
    if provider:
        params["custom_llm_provider"] = provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        params["api_key"] = os.environ["OPENAI_API_KEY"]
    if provider == "ollama":
        params.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))

    router = Router(model_list=[{"model_name": "scenario-parallel", "litellm_params": params}])

    requests = [
        RouterParallelRequest(
            model="scenario-parallel",
            messages=[{"role": "user", "content": PROMPTS[level]}],
        )
        for level in ("simple", "medium", "complex")
    ]

    async def run_parallel():
        results = await router.parallel_acompletions(
            requests,
            preserve_order=True,
            return_exceptions=True,
        )
        for item in results:
            if item.error:
                print(f"=== parallel completion #{item.index} ERROR ===\n{item.error}\n")
            else:
                print(f"=== parallel completion #{item.index} ({model_alias}) ===\n{item.content}\n")

    asyncio.run(run_parallel())


if __name__ == "__main__":
    main()
