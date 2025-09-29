#!/usr/bin/env python3
"""Run the LiteLLM mini-agent against a live LLM with increasing complexity."""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

try:
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        arun_mcp_mini_agent,
    )
except Exception as exc:  # pragma: no cover - runtime availability check
    print(f"Skipping mini-agent scenario (mini-agent components unavailable): {exc}")
    sys.exit(0)


def _infer_provider(alias: str) -> str | None:
    if alias.startswith("openai/"):
        return "openai"
    if alias.startswith("azure/"):
        return "azure_openai"
    if alias.startswith("gemini/"):
        return "gemini"
    if ":" in alias:
        return "ollama"
    return None


MODEL_ALIAS = (
    os.getenv("SCENARIO_MINI_MODEL")
    or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
    or os.getenv("LITELLM_DEFAULT_MODEL")
    or os.getenv("LITELLM_DEFAULT_LARGE_MODEL")
    or "ollama/qwen2.5-coder:14b"
)
PROVIDER = os.getenv("SCENARIO_MINI_PROVIDER") or _infer_provider(MODEL_ALIAS)

if PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
    print("Skipping mini-agent scenario (OPENAI_API_KEY not set)")
    sys.exit(0)

if PROVIDER == "ollama":
    os.environ.setdefault("OLLAMA_API_BASE", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))

PROMPTS: Dict[str, List[Dict[str, str]]] = {
    "simple": [
        {
            "role": "system",
            "content": (
                "You are a careful analyst. If a prompt involves arithmetic, you MUST run the exec_python tool "
                "to verify the answer before responding."
            ),
        },
        {
            "role": "user",
            "content": (
                "Compute (1729 / 7) + 48 - 13**2. Show the intermediate steps and only respond after running Python."
            ),
        },
    ],
    "medium": [
        {
            "role": "system",
            "content": (
                "You are an AI researcher with access to exec_python. Always run a Python check before giving a conclusion."
            ),
        },
        {
            "role": "user",
            "content": (
                "We observed sample temperatures in Celsius: [17.4, 19.8, 21.0, 22.4, 20.1]. "
                "Convert them to Fahrenheit, compute the average in both units, and explain what the change suggests."
            ),
        },
    ],
    "complex": [
        {
            "role": "system",
            "content": (
                "You are a meticulous research agent. Always cite interim computation results from exec_python before summarising."
            ),
        },
        {
            "role": "user",
            "content": (
                "A startup tracks weekly active users: [1320, 1588, 1710, 1895, 2044, 2102, 1998, 2234]. "
                "Using Python, compute the week-over-week percentage change, highlight the largest spike, and provide a recommendation."
            ),
        },
    ],
}

CONFIGS = {
    "simple": {"max_iterations": 3, "max_total_seconds": 60},
    "medium": {"max_iterations": 4, "max_total_seconds": 120},
    "complex": {"max_iterations": 5, "max_total_seconds": 150},
}


def _normalize_payload(data: Any) -> Any:
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data
    try:
        return data.model_dump()  # type: ignore[attr-defined]
    except Exception:
        try:
            return data.dict()  # type: ignore[attr-defined]
        except Exception:
            return str(data)


def _format_result(result, level: str) -> Dict[str, Any]:
    last_tool = None
    if result.iterations:
        last_iter = result.iterations[-1]
        if last_iter.tool_invocations:
            last_tool = last_iter.tool_invocations[-1]
    return {
        "level": level,
        "model": result.used_model,
        "iterations": len(result.iterations),
        "stopped_reason": result.stopped_reason,
        "final_answer": result.final_answer,
        "metrics": result.metrics,
        "last_tool_invocation": last_tool,
    }


def _agent_config(level: str) -> AgentConfig:
    cfg = CONFIGS[level]
    model_name = MODEL_ALIAS
    if "/" not in model_name and PROVIDER:
        model_name = f"{PROVIDER}/{model_name}"
    return AgentConfig(
        model=model_name,
        max_iterations=cfg["max_iterations"],
        max_total_seconds=cfg["max_total_seconds"],
        use_tools=True,
        auto_run_code_on_code_block=True,
        enable_repair=True,
        research_on_unsure=True,
    )


async def run_one(level: str) -> None:
    cfg = _agent_config(level)
    try:
        result = await arun_mcp_mini_agent(
            messages=PROMPTS[level],
            mcp=LocalMCPInvoker(shell_allow_prefixes=("python", "echo"), tool_timeout_sec=20.0),
            cfg=cfg,
        )
        payload = _format_result(result, level)
        print(f"=== mini-agent {level} ===")
        print(json.dumps(_normalize_payload(payload), indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"=== mini-agent {level} ERROR ===")
        print(str(exc))


async def main() -> None:
    for level in ("simple", "medium", "complex"):
        await run_one(level)


if __name__ == "__main__":
    asyncio.run(main())
