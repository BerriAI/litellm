#!/usr/bin/env python3
"""Live Router.parallel_acompletions demo hitting the configured provider."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from dotenv import find_dotenv, load_dotenv
from litellm import Router
from litellm.router_utils.parallel_acompletion import RouterParallelRequest

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
        print("No API key found for the configured model; skipping parallel release scenario.")
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

    return Router(model_list=[{"model_name": "release-parallel", "litellm_params": params}])


def format_response(resp) -> str:
    if hasattr(resp, "model_dump"):
        return json.dumps(resp.model_dump(), indent=2)  # type: ignore[call-arg]
    return json.dumps(resp, indent=2) if isinstance(resp, dict) else str(resp)


async def main_async() -> None:
    router = build_router()
    requests = [
        RouterParallelRequest("release-parallel", [{"role": "user", "content": "Ping from release scenario."}]),
        RouterParallelRequest("release-parallel", [{"role": "user", "content": "Provide a short tip about testing."}]),
    ]
    out = await router.parallel_acompletions(
        requests,
        preserve_order=True,
        return_exceptions=True,
        concurrency=2,
    )
    for item in out:
        payload = format_response(item.response) if item.error is None else f"ERROR: {item.error}"
        print(json.dumps({"index": item.index, "payload": payload}, indent=2))


def run() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
