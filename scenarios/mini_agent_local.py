#!/usr/bin/env python3
"""Run the mini-agent FastAPI shim with the local backend at multiple complexities."""

import asyncio
import json
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm.experimental_mcp_client.mini_agent.agent_proxy import AgentRunReq, run

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
    resp = await run(req)
    print(f"=== mini-agent {level} ===")
    print(json.dumps(resp, indent=2))

async def main() -> None:
    for level in ("simple", "medium", "complex"):
        await run_one(level)

if __name__ == "__main__":
    asyncio.run(main())
