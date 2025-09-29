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

MODEL_ALIAS = os.getenv("LITELLM_DEFAULT_CODE_MODEL") or "codex-agent/gpt-5"
litellm_params = {"model": MODEL_ALIAS}
api_key = os.getenv("CODEX_AGENT_API_KEY")
api_base = os.getenv("CODEX_AGENT_API_BASE")
if api_key:
    litellm_params["api_key"] = api_key
if api_base:
    litellm_params["api_base"] = api_base

model_list = [
    {
        "model_name": "codex-demo",
        "litellm_params": litellm_params,
    }
]

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
