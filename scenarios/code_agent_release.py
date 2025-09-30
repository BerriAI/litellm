#!/usr/bin/env python3
"""Invoke a tool-calling model and ensure a python_eval call is emitted."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from dotenv import find_dotenv, load_dotenv

import litellm

load_dotenv(find_dotenv())


def _resolve(name: str) -> str | None:
    value = os.getenv(name)
    if value and value.strip().startswith("${"):
        return None
    return value


def requires_api_key(model: str) -> bool:
    return not model.startswith("ollama/") and ":" not in model


def run_async() -> None:
    asyncio.run(main_async())


async def main_async() -> None:
    model = _resolve("CODE_AGENT_MODEL") or _resolve("LITELLM_DEFAULT_CODE_MODEL") or "openai/gpt-4o-mini"
    api_key = _resolve("CODE_AGENT_API_KEY")
    api_base = _resolve("CODE_AGENT_API_BASE")

    if requires_api_key(model) and not api_key:
        print("CODE_AGENT_API_KEY not set and model requires it; skipping code-agent scenario.")
        sys.exit(0)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "python_eval",
                "description": "Evaluate a simple arithmetic expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a code assistant."},
            {"role": "user", "content": "Use the tool to evaluate (2+3)*4 and show the result."},
        ],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "python_eval"}},
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    resp = await litellm.acompletion(**kwargs)
    payload = resp.model_dump() if hasattr(resp, "model_dump") else resp  # type: ignore[attr-defined]
    tool_calls = ((payload.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") if isinstance(payload, dict) else None
    print(json.dumps(payload, indent=2))
    if not tool_calls:
        raise RuntimeError("model did not produce tool_calls")


if __name__ == "__main__":
    run_async()
