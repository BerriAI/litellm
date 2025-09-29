#!/usr/bin/env python3
"""Single-call live mini-agent scenario using familiar LiteLLM patterns."""

import asyncio
import json
import os
import sys
from typing import List, Dict

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

try:
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        arun_mcp_mini_agent,
    )
except Exception as exc:  # pragma: no cover
    print(f"Mini-agent components unavailable: {exc}")
    sys.exit(1)

PROMPT: List[Dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are a meticulous analyst. Use exec_python to verify numbers before you answer."
        ),
    },
    {
        "role": "user",
        "content": (
            "A startup tracks weekly active users: [1320, 1588, 1710, 1895, 2044, 2102, 1998, 2234]. "
            "Using Python, compute the week-over-week percentage change, highlight the largest spike, "
            "and provide a recommendation."
        ),
    },
]

MODEL = (
    os.getenv("LITELLM_DEFAULT_CHUTES_MODEL")
    or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
    or "ollama/qwen2.5-coder:14b"
)

CONFIG = AgentConfig(
    model=MODEL,
    max_iterations=6,
    max_total_seconds=180,
    use_tools=True,
    auto_run_code_on_code_block=True,
)

async def main() -> None:
    print("-- mini-agent live scenario --")
    print(json.dumps({"model": CONFIG.model, "prompt": PROMPT}, indent=2))

    result = await arun_mcp_mini_agent(
        messages=PROMPT,
        mcp=LocalMCPInvoker(shell_allow_prefixes=("python", "echo"), tool_timeout_sec=30.0),
        cfg=CONFIG,
    )

    response_payload = {
        "final_answer": result.final_answer,
        "iterations": len(result.iterations),
        "stopped_reason": result.stopped_reason,
        "conversation": result.messages,
    }
    print(json.dumps({"request": PROMPT, "response": response_payload}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
