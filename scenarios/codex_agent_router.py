#!/usr/bin/env python3
"""Live codex-agent scenario using familiar Router setup."""

import asyncio
import json
import os
import sys
import time

from dotenv import find_dotenv, load_dotenv
from litellm import Router
load_dotenv(find_dotenv())

if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
    print("Codex-agent requires LITELLM_ENABLE_CODEX_AGENT=1; aborting.")
    sys.exit(1)

model_list = []

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    model_list.append(
        {
            "model_name": "gemini-2.5-flash",
            "litellm_params": {
                "model": "gemini/gemini-2.5-flash",
                "api_key": gemini_key,
            },
        }
    )
else:
    print("Skipping gemini scenario entry (GEMINI_API_KEY not set).")

chutes_key = os.getenv("CHUTES_API_KEY")
chutes_base = os.getenv("CHUTES_API_BASE")
chutes_model = os.getenv("LITELLM_DEFAULT_CHUTES_MODEL")
if chutes_key and chutes_base and chutes_model:
    chutes_params = {
        "model": chutes_model,
        "api_key": chutes_key,
        "api_base": chutes_base,
    }
    chutes_provider = os.getenv("SCENARIO_CHUTES_PROVIDER")
    if chutes_provider:
        chutes_params["custom_llm_provider"] = chutes_provider
    else:
        chutes_params["custom_llm_provider"] = "openai"
    model_list.append({"model_name": "deepseek-r1", "litellm_params": chutes_params})
else:
    print("Skipping chutes entry (set CHUTES_API_KEY, CHUTES_API_BASE, LITELLM_DEFAULT_CHUTES_MODEL to include it).")

codex_alias = os.getenv("LITELLM_DEFAULT_CODE_MODEL") or "codex-agent/gpt-5"


def infer_provider(alias: str) -> str | None:
    if alias.startswith("openai/"):
        return "openai"
    if alias.startswith("azure/"):
        return "azure_openai"
    if alias.startswith("ollama/"):
        return "ollama"
    if alias.startswith("codex-agent/"):
        return "codex-agent"
    if ":" in alias:
        return "ollama"
    return None


codex_provider = os.getenv("SCENARIO_CODE_PROVIDER") or infer_provider(codex_alias)
codex_params = {"model": codex_alias}
if codex_provider:
    codex_params["custom_llm_provider"] = codex_provider
if os.getenv("CODEX_AGENT_API_KEY"):
    codex_params["api_key"] = os.environ["CODEX_AGENT_API_KEY"]
if os.getenv("CODEX_AGENT_API_BASE"):
    codex_params["api_base"] = os.environ["CODEX_AGENT_API_BASE"]

if codex_alias.startswith("codex-agent/") and "api_base" not in codex_params:
    print("Skipping codex-agent scenario (CODEX_AGENT_API_BASE not configured for codex-agent/* model).")
    sys.exit(0)

model_list.append({"model_name": "codex-agent", "litellm_params": codex_params})

print("-- codex-agent scenario --")
print(json.dumps({"model_list": model_list}, indent=2))

router = Router(model_list=model_list)

PROMPTS = [
    {
        "level": "simple",
        "messages": [{"role": "user", "content": "Say hello then stop."}],
    },
    {
        "level": "medium",
        "messages": [{"role": "user", "content": "List three key features of LiteLLM."}],
    },
    {
        "level": "complex",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a planning agent with access to MCP tools (perplexity-ask, brave-search, context7). "
                    "Use the tools to gather evidence and respond in JSON with keys 'summary', 'insights', and 'sources'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Task list:\n"
                    "1. Research the latest LiteLLM features released in the past 3 months (prefer perplexity-ask).\n"
                    "2. Find public feedback or discussions about LiteLLM (use brave-search).\n"
                    "3. Locate the most relevant LiteLLM documentation for onboarding new developers (use context7 for 'liteLLM').\n"
                    "4. Recommend next steps for adopting LiteLLM in a production workflow.\n"
                    "Return JSON as instructed by the system message."
                ),
            }
        ],
    },
]


async def main() -> None:
    for prompt in PROMPTS:
        start = time.perf_counter()
        try:
            response = await router.acompletion(
                model="codex-agent",
                messages=prompt["messages"],
            )
            response_payload = response.model_dump() if hasattr(response, "model_dump") else str(response)
            duration = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "level": prompt["level"],
                        "request": prompt["messages"],
                        "response": response_payload,
                        "elapsed_s": round(duration, 2),
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "level": prompt["level"],
                        "request": prompt["messages"],
                        "error": str(exc),
                        "elapsed_s": round(duration, 2),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
