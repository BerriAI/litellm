#!/usr/bin/env python3
"""Live Chutes scenario using Router for an OpenAI-compatible model."""

from __future__ import annotations

import asyncio
import json
import os
import sys

from dotenv import find_dotenv, load_dotenv
from litellm import Router


load_dotenv(find_dotenv())

REQUIRED_ENV = ["CHUTES_API_KEY", "CHUTES_API_BASE"]


async def main_async() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        print(f"Skipping Chutes scenario (missing {', '.join(missing)}).")
        return

    model_alias = os.getenv("CHUTES_MODEL") or os.getenv("LITELLM_DEFAULT_MODEL") or "openai/deepseek-ai/DeepSeek-R1"

    router = Router(
        model_list=[
            {
                "model_name": "chutes",
                "litellm_params": {
                    "model": model_alias,
                    "api_key": os.environ["CHUTES_API_KEY"],
                    "api_base": os.environ["CHUTES_API_BASE"],
                },
            }
        ]
    )

    try:
        response = await router.acompletion(
            model="chutes",
            messages=[{"role": "user", "content": "Chutes release scenario ping."}],
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "401" in message or "Authentication" in message:
            print("Skipping Chutes scenario (authentication failed).")
            return
        if "404" in message or "No matching chute" in message:
            print("Skipping Chutes scenario (model not deployed at CHUTES_API_BASE).")
            return
        raise

    payload = response.model_dump() if hasattr(response, "model_dump") else response  # type: ignore[attr-defined]
    if isinstance(payload, dict):
        print(json.dumps(payload, indent=2))
    else:
        print(payload)


def run_async() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    run_async()
