#!/usr/bin/env python3
"""Run the mini-agent provider with Docker-backed tool execution."""

import asyncio
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from litellm import Router

PROMPTS = {
    "simple": [
        {
            "role": "user",
            "content": "Print Hello from inside the container using echo.",
        }
    ],
    "python": [
        {
            "role": "user",
            "content": (
                "Run this Python and return the output:\n"
                "```python\nprint('hello from docker')\n```"
            ),
        }
    ],
}


def main() -> None:
    if os.getenv("LITELLM_ENABLE_MINI_AGENT") != "1":
        print("Skipping mini-agent docker demo (LITELLM_ENABLE_MINI_AGENT != 1)")
        return

    container = os.getenv("LITELLM_MINI_AGENT_DOCKER_CONTAINER")
    if not container:
        print(
            "Set LITELLM_MINI_AGENT_DOCKER_CONTAINER to a running container name or ID. "
            "Start the bundled stack with `docker compose -f local/docker/compose.exec.yml up --build -d`."
        )
        return

    target_model = (
        os.getenv("SCENARIO_MINI_TARGET_MODEL")
        or os.getenv("LITELLM_DEFAULT_MODEL")
        or os.getenv("LITELLM_DEFAULT_CODE_MODEL")
        or "openai/gpt-4o-mini"
    )

    router = Router(
        model_list=[
            {
                "model_name": "mini-agent-docker",
                "litellm_params": {
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
                },
            }
        ]
    )

    async def run_all():
        for key in ("simple", "python"):
            try:
                response = await router.acompletion(model="mini-agent-docker", messages=PROMPTS[key])
                content = getattr(response.choices[0].message, "content", "")
                print(f"=== mini-agent-docker {key} ===\n{content}\n")
            except Exception as exc:
                print(f"=== mini-agent-docker {key} ERROR ===\n{exc}\n")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
