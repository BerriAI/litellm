#!/usr/bin/env python3
"""Live codex-agent scenario using familiar Router setup."""

import json
import os
import sys
import time

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

if os.getenv("LITELLM_ENABLE_CODEX_AGENT") != "1":
    print("Codex-agent requires LITELLM_ENABLE_CODEX_AGENT=1; aborting.")
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
                "role": "user",
                "content": "Draft a short proposal outlining how LiteLLM can integrate with an existing FastAPI application, including pros and cons.",
            }
        ],
    },
]

for prompt in PROMPTS:
    start = time.perf_counter()
    response = router.completion(model="codex-demo", messages=prompt["messages"])
    duration = time.perf_counter() - start
    content = getattr(response.choices[0].message, "content", "").strip()
    print(
        json.dumps(
            {
                "level": prompt["level"],
                "request": prompt["messages"],
                "response": content,
                "elapsed_s": round(duration, 2),
            },
            indent=2,
        )
    )
