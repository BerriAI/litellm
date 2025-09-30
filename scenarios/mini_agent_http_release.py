#!/usr/bin/env python3
"""Live mini-agent HTTP API check hitting /agent/run."""

from __future__ import annotations

import json
import os
import sys

import httpx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def _resolve(name: str) -> str | None:
    value = os.getenv(name)
    if value and value.strip().startswith("${"):
        return None
    return value


def resolve_model() -> str:
    return (
        _resolve("MINI_AGENT_MODEL")
        or _resolve("LITELLM_DEFAULT_MODEL")
        or "openai/gpt-4o-mini"
    )


def run() -> None:
    base = _resolve("MINI_AGENT_URL")
    if not base:
        print("MINI_AGENT_URL not set; skipping mini-agent HTTP release scenario.")
        sys.exit(0)
    url = f"{base.rstrip('/')}/agent/run"
    payload = {
        "messages": [{"role": "user", "content": "Release scenario ping"}],
        "model": resolve_model(),
        "tool_backend": "local",
    }
    print(f"POST {url}\nPayload: {json.dumps(payload, indent=2)}")
    response = httpx.post(url, json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    print("Response:", json.dumps(data, indent=2))
    if not data.get("ok"):
        raise RuntimeError("mini-agent API responded with ok=False")
    final = data.get("final_answer") or "".join(
        m.get("content", "") for m in data.get("messages", []) if m.get("role") == "assistant"
    )
    if not final.strip():
        raise RuntimeError("assistant response missing from mini-agent API")


if __name__ == "__main__":
    run()
