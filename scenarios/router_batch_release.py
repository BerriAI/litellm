#!/usr/bin/env python3
"""Live Router.abatch_completion scenario for release validation."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from dotenv import find_dotenv, load_dotenv
from litellm import Router

load_dotenv(find_dotenv())


def _resolve(name: str) -> str | None:
    value = os.getenv(name)
    if value and value.strip().startswith("${"):
        return None
    return value


def infer_provider(model: str) -> str | None:
    if model.startswith("ollama/") or ":" in model:
        return "ollama"
    if model.startswith("openai/"):
        return "openai"
    if model.startswith("azure/"):
        return "azure_openai"
    if model.startswith("gemini/"):
        return "gemini"
    if model.startswith("anthropic/") or model.startswith("claude/"):
        return "anthropic"
    return None


def build_router() -> Router:
    model = _resolve("RELEASE_PARALLEL_MODEL") or _resolve("LITELLM_DEFAULT_MODEL") or "openai/gpt-4o-mini"
    provider = infer_provider(model)
    api_key = _resolve("RELEASE_PARALLEL_API_KEY") or _resolve("OPENAI_API_KEY")

    if provider != "ollama" and not api_key:
        print("No API key found for the configured model; skipping batch release scenario.")
        sys.exit(0)

    params: dict[str, str] = {"model": model}
    if provider:
        params["custom_llm_provider"] = provider
    if api_key:
        params["api_key"] = api_key

    if provider == "ollama":
        params.setdefault("api_base", _resolve("OLLAMA_API_BASE") or "http://127.0.0.1:11434")
    else:
        api_base = _resolve("RELEASE_PARALLEL_API_BASE") or _resolve("OPENAI_API_BASE")
        if api_base:
            params["api_base"] = api_base

    return Router(model_list=[{"model_name": "release-batch", "litellm_params": params}])


async def main_async() -> None:
    router = build_router()
    models = ["release-batch", "release-batch"]
    messages = [
        [{"role": "user", "content": "Summarize the benefits of unit testing."}],
        [{"role": "user", "content": "Provide a short onboarding tip."}],
    ]
    out = await router.abatch_completion(models=models, messages=messages)
    for row in out:
        for resp in row:
            payload = resp.model_dump() if hasattr(resp, "model_dump") else resp  # type: ignore[attr-defined]
            print(json.dumps(payload, indent=2))


def run() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
