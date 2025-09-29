#!/usr/bin/env python3
"""Live codex-agent scenario using familiar Router setup."""

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
