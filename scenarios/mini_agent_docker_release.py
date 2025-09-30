#!/usr/bin/env python3
"""Validate the mini-agent Docker backend is reachable and executes tools."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

from dotenv import find_dotenv, load_dotenv

from litellm import Router
from litellm.extras import clean_json_string

load_dotenv(find_dotenv())

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a structured analyst. Use exec_python to run code inside the Docker container. "
        "Respond with a JSON summary when done."
    ),
}

USER_PROMPT = {
    "role": "user",
    "content": (
        "Run this Python exactly and summarise:"
        "```python\n"
        "import platform\n"
        "import os\n"
        "print({'cwd': os.getcwd(), 'py': platform.python_version()})\n"
        "```"
    ),
}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Environment variable {name} must be set for this scenario.")
        sys.exit(1)
    return value


def _ensure_container_running(container: str) -> None:
    try:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("Docker CLI not found; install Docker or adjust the container backend.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or exc.stdout.strip() or str(exc))
        print(f"Container '{container}' is not accessible.")
        sys.exit(1)

    if proc.stdout.strip().lower() != "true":
        print(f"Container '{container}' is not running (inspect returned {proc.stdout.strip()!r}).")
        sys.exit(1)


def build_router(container: str) -> Router:
    target_model = (
        os.getenv("SCENARIO_MINI_TARGET_MODEL")
        or os.getenv("LITELLM_DEFAULT_MODEL")
        or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
        or "openai/gpt-4o-mini"
    )

    params = {
        "model": "mini-agent",
        "custom_llm_provider": "mini-agent",
        "target_model": target_model,
        "tool_backend": "docker",
        "docker_container": container,
        "allowed_languages": os.getenv("LITELLM_MINI_AGENT_LANGUAGES", "python,rust,go,javascript"),
        "max_iterations": int(os.getenv("SCENARIO_MINI_MAX_ITER", "4")),
        "max_seconds": float(os.getenv("SCENARIO_MINI_MAX_SECONDS", "60")),
        "temperature": 0,
        "tool_choice": "required",
        "response_format": {"type": "json_object"},
        "seed": 7,
    }

    return Router(model_list=[{"model_name": "mini-agent-docker", "litellm_params": params}])


async def main_async() -> None:
    if os.getenv("LITELLM_ENABLE_MINI_AGENT") != "1":
        print("Set LITELLM_ENABLE_MINI_AGENT=1 before running this scenario.")
        sys.exit(1)

    container = _require_env("LITELLM_MINI_AGENT_DOCKER_CONTAINER")
    _ensure_container_running(container)

    router = build_router(container)
    response = await router.acompletion(
        model="mini-agent-docker",
        messages=[SYSTEM_PROMPT, USER_PROMPT],
    )

    payload = response.model_dump() if hasattr(response, "model_dump") else json.loads(json.dumps(response))
    convo = payload.get("additional_kwargs", {}).get("mini_agent", {}).get("conversation", [])
    summary = payload.get("additional_kwargs", {}).get("mini_agent", {}).get("parsed_tools", [])

    print(json.dumps({
        "conversation": convo,
        "parsed_tools": summary,
    }, indent=2, default=str))

    if not summary:
        print("parsed_tools empty – mini-agent did not record tool output.")
        sys.exit(1)


def run() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
