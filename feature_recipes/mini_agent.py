"""Mini-agent feature recipe.

This script showcases the paved-road usage of the LiteLLM mini-agent helper:

    1. Configure the target model (defaults to ``LITELLM_DEFAULT_CHUTES_MODEL``).
    2. Whitelist tool languages via ``tools=("python", "rust")``.
    3. Provide OpenAI-style ``messages`` with system/user roles.
    4. Await the final answer and inspect iteration counts.

Run this module directly (``python feature_recipes/mini_agent.py``) after
populating the relevant environment variables.
"""

import asyncio
import os
from typing import Iterable, List, Dict

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
    AgentConfig,
    LocalMCPInvoker,
    arun_mcp_mini_agent,
)

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


def run_mini_agent(
    messages: List[Dict[str, str]],
    *,
    model: str,
    max_iterations: int = 6,
    max_seconds: float = 180.0,
    tools: Iterable[str] = ("python",),
) -> Dict[str, object]:
    """Helper that keeps the paved-road mini-agent call ergonomic."""

    async def _call() -> Dict[str, object]:
        cfg = AgentConfig(
            model=model,
            max_iterations=max_iterations,
            max_total_seconds=max_seconds,
            use_tools=True,
            auto_run_code_on_code_block=True,
        )
        invoker = LocalMCPInvoker(
            shell_allow_prefixes=tuple(tools),
            tool_timeout_sec=max_seconds,
        )
        result = await arun_mcp_mini_agent(messages=messages, mcp=invoker, cfg=cfg)
        return {
            "final_answer": result.final_answer,
            "iterations": len(result.iterations),
            "stopped_reason": result.stopped_reason,
        }

    return asyncio.run(_call())


def main() -> None:
    model = (
        os.getenv("LITELLM_DEFAULT_CHUTES_MODEL")
        or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
        or "ollama/qwen2.5-coder:14b"
    )
    print({"model": model, "prompt": PROMPT})
    result = run_mini_agent(PROMPT, model=model, tools=("python", "rust"))
    print(result)


if __name__ == "__main__":
    main()
