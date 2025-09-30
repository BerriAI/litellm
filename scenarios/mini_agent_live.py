#!/usr/bin/env python3
"""Live mini-agent scenario via Router using the mini-agent provider."""

import asyncio
import json
import os
import sys
import ast

from dotenv import find_dotenv, load_dotenv
from litellm import Router
from litellm.extras import clean_json_string

load_dotenv(find_dotenv())

if os.getenv("LITELLM_ENABLE_MINI_AGENT") != "1":
    print("Mini-agent requires LITELLM_ENABLE_MINI_AGENT=1; aborting.")
    sys.exit(1)

BASE_MODEL = (
    os.getenv("SCENARIO_MINI_TARGET_MODEL")
    or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
    or os.getenv("LITELLM_DEFAULT_MODEL")
    or "openai/gpt-4o-mini"
)

if BASE_MODEL.startswith("openai/") and not os.getenv("OPENAI_API_KEY"):
    print("Skipping mini-agent live scenario (OPENAI_API_KEY not set for openai base model).")
    sys.exit(0)

model_list = [
    {
        "model_name": "mini-agent",
        "litellm_params": {
            "model": "mini-agent",
            "custom_llm_provider": "mini-agent",
            "target_model": BASE_MODEL,
            "allowed_languages": ["python"],
            "max_iterations": 4,
            "max_seconds": 180,
            "temperature": 0,
            "tool_choice": "required",
            "response_format": {"type": "json_object"},
            "seed": 7,
        },
    }
]

router = Router(model_list=model_list)

PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a meticulous analyst. You may use the tool `exec_python` to run short Python snippets. "
            "**Never** invent new tool names. If the calculation is complete, respond with a concise natural-language summary."
        ),
    },
    {
        "role": "user",
        "content": (
            "A startup tracks weekly active users: [1320, 1588, 1710, 1895, 2044, 2102, 1998, 2234].\n"
            "Run the following Python exactly as provided, then summarise the largest spike and give a short recommendation.\n"
            "```python\n"
            "users = [1320, 1588, 1710, 1895, 2044, 2102, 1998, 2234]\n"
            "changes = [((curr - prev) / prev) * 100 for prev, curr in zip(users, users[1:])]\n"
            "peak_change = max(changes)\n"
            "peak_week = changes.index(peak_change) + 2  # week numbering starts at 1\n"
            "print({'changes': changes, 'peak_week': peak_week, 'peak_change': peak_change})\n"
            "```\n"
            "After the code runs, respond with one sentence that states the peak percentage change, the week number, and a concrete recommendation."
        ),
    },
]


async def main() -> None:
    print(json.dumps({"model_list": model_list, "prompt": PROMPT}, indent=2))
    try:
        response = await router.acompletion(model="mini-agent", messages=PROMPT)
        payload = response.model_dump() if hasattr(response, "model_dump") else str(response)
        print(json.dumps({"request": PROMPT, "response": payload}, indent=2))

        # Try to build a deterministic summary from the tool output so we are not at the
        # mercy of the base model's final wording (many Ollama models return "{}" here).
        summary = None
        mini_meta = getattr(response, "additional_kwargs", {}).get("mini_agent", {}) if hasattr(response, "additional_kwargs") else {}
        for message in reversed(mini_meta.get("conversation", [])):
            if isinstance(message, dict) and message.get("role") == "tool":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    parsed = None
                    if clean_json_string is not None:
                        try:
                            parsed = clean_json_string(content.strip(), return_dict=True)
                        except Exception:
                            parsed = None
                    if parsed is None:
                        try:
                            parsed = ast.literal_eval(content.strip())
                        except Exception:
                            parsed = None
                    if isinstance(parsed, dict):
                        peak_change = parsed.get("peak_change")
                        peak_week = parsed.get("peak_week")
                        if peak_change is not None and peak_week is not None:
                            summary = {
                                "peak_change_pct": round(float(peak_change), 1),
                                "peak_week": int(peak_week),
                                "recommendation": "Investigate the drivers to sustain momentum.",
                            }
                            break
        if summary is None:
            summary = {}
        print(json.dumps({"synthetic_summary": summary}, indent=2))

        finish_reason = None
        if getattr(response, "choices", None):
            finish_reason = getattr(response.choices[0], "finish_reason", None)

        if finish_reason != "success":
            stopped = mini_meta.get("stopped_reason") if mini_meta else None
            print(
                "Recommendation: mini-agent stopped early (reason: {}). "
                "Use a stronger base model via SCENARIO_MINI_TARGET_MODEL or increase SCENARIO_MINI_MAX_ITER.".format(
                    stopped or finish_reason or "unknown"
                )
            )
    except Exception as exc:
        print(json.dumps({"request": PROMPT, "error": str(exc)}, indent=2))
        msg = str(exc)
        if "model '" in msg and "not found" in msg:
            print(
                "Recommendation: pull the requested Ollama model (e.g. `ollama pull glm4:12b`) "
                "or point SCENARIO_MINI_TARGET_MODEL at an installed model."
            )
        else:
            print(
                "Recommendation: use a larger base model (e.g. openai/gpt-4o-mini) or relax SCENARIO_MINI_MAX_ITER."
            )


if __name__ == "__main__":
    asyncio.run(main())
