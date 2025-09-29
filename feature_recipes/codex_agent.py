"""Feature recipe: calling codex-agent through the LiteLLM Router."""

import json
import os
import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
    print("Set LITELLM_ENABLE_CODEX_AGENT=1 before running this recipe.")
    sys.exit(1)

from litellm import Router

MODEL_ALIAS = os.getenv("SCENARIO_CODE_MODEL") or "codex-agent/gpt-5"

model_list = [
    {
        "model_name": "codex-demo",
        "litellm_params": {
            "model": MODEL_ALIAS,
            **({"api_key": os.environ["CODEX_AGENT_API_KEY"]} if os.getenv("CODEX_AGENT_API_KEY") else {}),
            **({"api_base": os.environ["CODEX_AGENT_API_BASE"]} if os.getenv("CODEX_AGENT_API_BASE") else {}),
        },
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
print(json.dumps({"content": response.choices[0].message.content}, indent=2))
