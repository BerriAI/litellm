"""Codex-agent feature recipe.

Demonstrates invoking the codex-agent provider via the standard LiteLLM Router:

- Uses ``model_list`` with a codex-agent entry.
- Reads API base/key from ``CODEX_AGENT_API_BASE`` / ``CODEX_AGENT_API_KEY``.
- Issues a single completion request with OpenAI-style messages.

Run ``python feature_recipes/codex_agent.py`` with ``LITELLM_ENABLE_CODEX_AGENT=1``
and the codex CLI/sidecar configured.
"""

import json
import os
import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
    print("Set LITELLM_ENABLE_CODEX_AGENT=1 before running this recipe.")
    sys.exit(1)

from litellm import Router

gemini_api_key = os.getenv("GEMINI_API_KEY")
model_list = [
    {
        "model_name": "gemini-2.5-flash",
        "litellm_params": {
            "model": "gemini/gemini-2.5-flash",
            "api_key": gemini_api_key,
        }
        if gemini_api_key
        else {"model": "gemini/gemini-2.5-flash"},
    },
    {
        "model_name": "deepseek-r1",
        "litellm_params": {
            "model": os.getenv("LITELLM_DEFAULT_CHUTES_MODEL") or "deepseek-ai/DeepSeek-R1",
            "api_key": os.getenv("CHUTES_API_KEY"),
            "api_base": os.getenv("CHUTES_API_BASE"),
        },
    },
]

codex_alias = os.getenv("LITELLM_DEFAULT_CODE_MODEL") or "codex-agent/gpt-5"
codex_params = {"model": codex_alias}
if os.getenv("CODEX_AGENT_API_KEY"):
    codex_params["api_key"] = os.environ["CODEX_AGENT_API_KEY"]
if os.getenv("CODEX_AGENT_API_BASE"):
    codex_params["api_base"] = os.environ["CODEX_AGENT_API_BASE"]

model_list.append({"model_name": "codex-demo", "litellm_params": codex_params})

router = Router(model_list=model_list)

PROMPT = [
    {
        "role": "user",
        "content": "List three key features of LiteLLM.",
    }
]

print(json.dumps({"model_list": model_list, "prompt": PROMPT}, indent=2))
response = router.completion(model="codex-demo", messages=PROMPT)
print(
    json.dumps(
        {
            "request": PROMPT,
            "response": response.choices[0].message.content,
        },
        indent=2,
    )
)
