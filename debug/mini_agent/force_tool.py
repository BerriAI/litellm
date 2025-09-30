#!/usr/bin/env python3
import asyncio
import json
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router

BASE_MODEL = os.getenv("DEBUG_MINI_AGENT_MODEL") or os.getenv("SCENARIO_MINI_TARGET_MODEL")
if not BASE_MODEL:
    raise SystemExit("Set DEBUG_MINI_AGENT_MODEL or SCENARIO_MINI_TARGET_MODEL")

MAX_ITER = int(os.getenv("DEBUG_MINI_AGENT_MAX_ITER", "4"))

router = Router(
    model_list=[
        {
            "model_name": "debug-mini-agent",
            "litellm_params": {
                "model": "mini-agent",
                "custom_llm_provider": "mini-agent",
                "target_model": BASE_MODEL,
                "allowed_languages": ["python"],
                "max_iterations": MAX_ITER,
                "max_seconds": 180,
                "temperature": float(os.getenv("DEBUG_MINI_AGENT_TEMPERATURE", "0")),
                "tool_choice": os.getenv("DEBUG_MINI_AGENT_TOOL_CHOICE", "required"),
                "response_format": json.loads(os.getenv("DEBUG_MINI_AGENT_RESPONSE_FORMAT", '{"type":"json_object"}')),
                "seed": int(os.getenv("DEBUG_MINI_AGENT_SEED", "7")),
            },
        }
    ]
)

PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a coding assistant. You may call the tool `exec_python` to run Python snippets. "
            "Always call the tool when asked to compute something; do not reply until the code has been executed."
        ),
    },
    {
        "role": "user",
        "content": (
            "Run `exec_python` to compute the sum of the numbers [1, 2, 3, 4] and then reply with the result."
        ),
    },
]

async def main() -> None:
    resp = await router.acompletion(model="debug-mini-agent", messages=PROMPT)
    if hasattr(resp, "model_dump"):
        payload = resp.model_dump()
    else:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": getattr(resp.choices[0].message, "content", None)
                    }
                }
            ]
        }
    print(json.dumps(payload, indent=2))
    extra = getattr(resp, "additional_kwargs", {})
    if extra:
        print("\nadditional_kwargs:")
        try:
            print(json.dumps(extra, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))
        except TypeError:
            print(str(extra))

if __name__ == "__main__":
    asyncio.run(main())
