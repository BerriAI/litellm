#!/usr/bin/env python3
"""Run the mini-agent FastAPI shim with the local backend at multiple complexities."""

import asyncio
import json
import sys
from typing import Any
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

try:
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import AgentRunReq, run
except Exception as exc:
    print(f"Skipping mini-agent scenario (mini-agent components unavailable): {exc}")
    sys.exit(0)

PROMPTS = {
    "simple": [{"role": "user", "content": "Say hi"}],
    "medium": [{"role": "user", "content": "Walk me through two steps to brew coffee."}],
    "complex": [
        {"role": "system", "content": "You are a meticulous research assistant. Provide thorough answers with bullet points when helpful."},
        {"role": "user", "content": "Summarize the differences between supervised and unsupervised learning in a paragraph."},
    ],
}

CONFIGS = {
    "simple": {"tool_backend": "local", "max_iterations": 1, "max_total_seconds": 10},
    "medium": {"tool_backend": "local", "max_iterations": 3, "max_total_seconds": 30},
    "complex": {"tool_backend": "local", "max_iterations": 5, "max_total_seconds": 60},
}

async def run_one(level: str) -> None:
    cfg = CONFIGS[level]
    req = AgentRunReq(
        messages=PROMPTS[level],
        model=f"mini-agent-{level}",
        **cfg,
    )
    try:
        resp = await run(req)
        payload: Any = resp
        try:
            payload = resp.model_dump()  # type: ignore[attr-defined]
        except Exception:
            try:
                payload = resp.dict()  # type: ignore[attr-defined]
            except Exception:
                if not isinstance(resp, (dict, list, str, int, float, bool, type(None))):
                    payload = str(resp)
        print(f"=== mini-agent {level} ===")
        print(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"=== mini-agent {level} ERROR ===")
        print(str(exc))

async def main() -> None:
    for level in ("simple", "medium", "complex"):
        await run_one(level)

if __name__ == "__main__":
    asyncio.run(main())
