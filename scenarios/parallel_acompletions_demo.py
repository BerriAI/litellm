#!/usr/bin/env python3
"""Demonstrate Router.parallel_acompletions with increasing prompt complexity."""

import asyncio
import json
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router
try:
    from litellm.router_utils.parallel_acompletion import RouterParallelRequest
except Exception:
    from dataclasses import dataclass
    from typing import Any, List, Dict, Optional

    @dataclass
    class RouterParallelRequest:  # type: ignore
        model: str
        messages: List[Dict[str, Any]]
        temperature: Optional[float] = None
        max_tokens: Optional[int] = None
        top_p: Optional[float] = None
        stream: Optional[bool] = None
        kwargs: Optional[Dict[str, Any]] = None

PROMPTS = {
    "simple": "List three programming languages.",
    "medium": "Provide a fun fact about pandas and mention their habitat.",
    "complex": "Summarize the benefits and trade-offs of unit testing versus integration testing in a short paragraph.",
}


def infer_provider(alias: str) -> str | None:
    if alias.startswith("openai/"):
        return "openai"
    if alias.startswith("azure/"):
        return "azure_openai"
    if alias.startswith("gemini/"):
        return "gemini"
    if ":" in alias or alias.startswith("ollama/"):
        return "ollama"
    return None


def main() -> None:
    model_alias = (
        os.getenv("SCENARIO_PARALLEL_MODEL")
        or os.getenv("LITELLM_DEFAULT_MODEL")
        or "ollama/glm4:latest"
    )

    provider = os.getenv("SCENARIO_PARALLEL_PROVIDER") or infer_provider(model_alias)

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY not set; falling back to local Ollama model 'ollama/glm4:latest'."
        )
        model_alias = "ollama/glm4:latest"
        provider = "ollama"

    params = {"model": model_alias}
    if provider:
        params["custom_llm_provider"] = provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        params["api_key"] = os.environ["OPENAI_API_KEY"]
    if provider == "ollama":
        params.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))

    router = Router(model_list=[{"model_name": "scenario-parallel", "litellm_params": params}])

    messages_list = [
        [{"role": "user", "content": PROMPTS["simple"]}],
        [{"role": "user", "content": PROMPTS["medium"]}],
        [{"role": "user", "content": PROMPTS["complex"]}],
        [{"role": "user", "content": "Give two tips for onboarding a new engineer."}],
    ]

    requests = [
        RouterParallelRequest("scenario-parallel", messages_list[0], temperature=0.3),
        RouterParallelRequest("scenario-parallel", messages_list[1], stream=True),
        {
            "model": "scenario-parallel",
            "messages": messages_list[2],
            "kwargs": {"max_tokens": 128},
        },
        RouterParallelRequest("scenario-parallel", messages_list[3], top_p=0.9),
    ]

    async def run_parallel():
        results = await router.parallel_acompletions(
            requests,
            preserve_order=True,
            return_exceptions=True,
            concurrency=2,
        )
        for item in results:
            response_payload = getattr(item.response, "model_dump", None)
            if callable(response_payload):
                response_payload = response_payload()
            else:
                response_payload = item.response
            print(
                json.dumps(
                    {
                        "index": item.index,
                        "request": item.request.messages,
                        "response": response_payload,
                        "content": item.content,
                        "error": str(item.error) if item.error else None,
                    },
                    indent=2,
                )
            )

    asyncio.run(run_parallel())


if __name__ == "__main__":
    main()
