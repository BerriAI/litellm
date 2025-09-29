#!/usr/bin/env python3
"""Route requests through the default code model at multiple prompt complexities."""
from __future__ import annotations

import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router

PROMPTS = {
    "simple": "Say hello then stop.",
    "medium": "List three key features of LiteLLM.",
    "complex": "Draft a short proposal outlining how LiteLLM can integrate with an existing FastAPI application, including pros and cons.",
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
        os.getenv("SCENARIO_CODE_MODEL")
        or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
        or "ollama/qwen2.5-coder:14b"
    )

    provider = os.getenv("SCENARIO_CODE_PROVIDER") or infer_provider(model_alias)

    params = {"model": model_alias}
    if provider:
        params["custom_llm_provider"] = provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        params["api_key"] = os.environ["OPENAI_API_KEY"]
    if provider == "ollama":
        params.setdefault("api_base", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))

    if model_alias.startswith("codex-agent/"):
        if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
            print("Skipping codex-agent scenario (LITELLM_ENABLE_CODEX_AGENT != 1)")
            return
        params.setdefault("api_base", os.getenv("CODEX_AGENT_API_BASE"))
        if not params.get("api_base"):
            print("Skipping codex-agent scenario (CODEX_AGENT_API_BASE not set)")
            return

    router = Router(model_list=[{"model_name": "scenario-code", "litellm_params": params}])

    for level in ("simple", "medium", "complex"):
        response = router.completion(
            model="scenario-code",
            messages=[{"role": "user", "content": PROMPTS[level]}],
        )
        content = getattr(response.choices[0].message, "content", "").strip()
        print(f"=== code-model ({model_alias}) {level} ===\n{content}\n")


if __name__ == "__main__":
    main()
