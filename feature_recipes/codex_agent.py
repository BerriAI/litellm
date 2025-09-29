"""Codex-agent feature recipe.

Demonstrates invoking the codex-agent provider via the standard LiteLLM Router:

- Uses ``model_list`` with a codex-agent entry.
- Reads API base/key from ``CODEX_AGENT_API_BASE`` / ``CODEX_AGENT_API_KEY``.
- Issues a single completion request with OpenAI-style messages.

Run ``python feature_recipes/codex_agent.py`` with ``LITELLM_ENABLE_CODEX_AGENT=1``
and the codex CLI/sidecar configured.
"""

import asyncio
import json
import os
import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
    print("Set LITELLM_ENABLE_CODEX_AGENT=1 before running this recipe.")
    sys.exit(1)

from litellm import Router


model_list = [
    {
        "model_name": "gemini-2.5-flash",
        "litellm_params": {
            "model": "gemini/gemini-2.5-flash",
            "api_key": os.getenv("GEMINI_API_KEY"),
        }
    },
    {
        "model_name": "deepseek-r1",
        "litellm_params": {
            "model": os.getenv("LITELLM_DEFAULT_CHUTES_MODEL") or "deepseek-ai/DeepSeek-R1",
            "api_key": os.getenv("CHUTES_API_KEY"),
            "api_base": os.getenv("CHUTES_API_BASE"),
        },
    },
    {
        "model_name": "codex-agent",
        "litellm_params": {
            "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL") or "codex-agent/gpt-5",
            "api_key": os.getenv("CODEX_AGENT_API_KEY"),
            "api_base": os.getenv("CODEX_AGENT_API_BASE"),
        },
    },
]

router = Router(model_list=model_list)

PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a planning agent with access to MCP tools (perplexity-ask, brave-search, context7). "
            "For each task decide which tool to call and cite results. Respond in JSON with keys "
            "'summary', 'insights', and 'sources'."
        ),
    },
    {
        "role": "user",
        "content": (
            "Task list:\n"
            "1. Research the latest features shipped in LiteLLM within the past 3 months (try perplexity-ask).\n"
            "2. Find community feedback or discussions about LiteLLM (use brave-search).\n"
            "3. Look up the most relevant LiteLLM documentation pages for onboarding (use context7 for 'liteLLM').\n"
            "4. Provide recommended next steps for adopting LiteLLM in production.\n"
            "Return JSON matching the system instructions."
        ),
    },
]


async def main() -> None:
    print(json.dumps({"model_list": model_list, "prompt": PROMPT}, indent=2))
    response = await router.acompletion(model="codex-agent", messages=PROMPT)
    response_payload = response.model_dump() if hasattr(response, "model_dump") else str(response)
    print(
        json.dumps(
            {
                "request": PROMPT,
                "response": response_payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
