#!/usr/bin/env python3
"""Live codex-agent scenario using familiar Router setup."""

import json
import os
import sys
import time

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

CODex_FLAG = os.getenv("LITELLM_ENABLE_CODEX_AGENT")
if CODex_FLAG != "1":
    print("Codex-agent requires LITELLM_ENABLE_CODEX_AGENT=1; aborting.")
    sys.exit(1)

from litellm import Router

PROMPTS = {
    "simple": "Say hello then stop.",
    "medium": "List three key features of LiteLLM.",
    "complex": "Draft a short proposal outlining how LiteLLM can integrate with an existing FastAPI application, including pros and cons.",
}

MODEL_ALIAS = os.getenv("SCENARIO_CODE_MODEL") or "codex-agent/gpt-5"

params = {"model": MODEL_ALIAS}
if os.getenv("CODEX_AGENT_API_KEY"):
    params["api_key"] = os.environ["CODEX_AGENT_API_KEY"]
if os.getenv("CODEX_AGENT_API_BASE"):
    params["api_base"] = os.environ["CODEX_AGENT_API_BASE"]

print("-- codex-agent scenario --")
print(json.dumps({"model_list": [{"model_name": "codex-demo", "litellm_params": params}]}, indent=2))

router = Router(model_list=[{"model_name": "codex-demo", "litellm_params": params}])

for level in ("simple", "medium", "complex"):
    messages = [{"role": "user", "content": PROMPTS[level]}]
    print(json.dumps({"level": level, "messages": messages}, indent=2))
    start = time.perf_counter()
    response = router.completion(model="codex-demo", messages=messages)
    duration = time.perf_counter() - start
    content = getattr(response.choices[0].message, "content", "").strip()
    print(f"=== codex-agent {level} (elapsed {duration:.2f}s) ===\n{content}\n")
