"""CodeWorld + LiteLLM feature recipe (interface sketch).

Goal: show a *simple, interchangeable* Router call that kicks off a CodeWorld
run. The call mirrors normal LiteLLM usage—supply a prompt, metrics, allowed
languages, iterations, and let the provider take up to a few minutes to return
aggregated results.
"""

from __future__ import annotations

import asyncio
import json
import os
from textwrap import dedent

from dotenv import find_dotenv, load_dotenv

from litellm import Router
from .codeworld_provider import CodeWorldProvider  # local adapter

load_dotenv(find_dotenv())

# --- Tunables (env overrides keep the interface configurable) -----------------
PROMPT = os.getenv(
    "CODEWORLD_PROMPT",
    dedent(
        """
        Compare the provided multiplication strategies and recommend the best
        option for production use. Highlight correctness, robustness, speed,
        and brevity. Include concrete next steps for the team.
        """
    ).strip(),
)

CODEWORLD_METRICS = [
    metric.strip()
    for metric in os.getenv(
        "CODEWORLD_METRICS",
        "correctness, robustness, speed, brevity",
    ).split(",")
    if metric.strip()
]

ALLOWED_LANGUAGES = [
    lang.strip()
    for lang in os.getenv(
        "CODEWORLD_ALLOWED_LANGUAGES",
        "python, rust, go, javascript",
    ).split(",")
    if lang.strip()
]

MAX_ITERATIONS = int(os.getenv("CODEWORLD_MAX_ITERATIONS", "3"))
TEMPERATURE = float(os.getenv("CODEWORLD_TEMPERATURE", "0"))
SEED = int(os.getenv("CODEWORLD_SEED", "37"))
MAX_SECONDS = float(os.getenv("CODEWORLD_TIMEOUT_SECONDS", "300"))  # CodeWorld runs are long-lived

TARGET_MODEL = (
    os.getenv("CODEWORLD_TARGET_MODEL")
    or os.getenv("LITELLM_DEFAULT_MODEL")
    or "ollama/qwen2.5-coder:14b"
)

# --- Router configuration -----------------------------------------------------
# In a real integration, register CodeWorldProvider with LiteLLM's provider registry.
# Here we keep a local adapter for clarity in the feature recipe.
CODEWORLD_BASE = os.getenv("CODEWORLD_BASE", "http://localhost:8000")
CODEWORLD_TOKEN = os.getenv("CODEWORLD_TOKEN")
codeworld_provider = CodeWorldProvider(base=CODEWORLD_BASE, token=CODEWORLD_TOKEN)

messages = [
    {
        "role": "system",
        "content": (
            "You orchestrate CodeWorld runs. Use the provided metrics to score variants"
            " and respond with a JSON object containing keys: summary, winning_strategy,"
            " metrics_breakdown, next_steps."
        ),
    },
    {
        "role": "user",
        "content": PROMPT,
    },
]


async def main() -> None:
    print(json.dumps(
        {
            "example_request": {
                "model": "codeworld",
                "messages": messages,
                "codeworld_metrics": CODEWORLD_METRICS,
                "codeworld_iterations": MAX_ITERATIONS,
                "codeworld_allowed_languages": ALLOWED_LANGUAGES,
                "codeworld_timeout_seconds": MAX_SECONDS,
            }
        },
        indent=2,
    ))

    # Feature recipe sketch: this is the single LiteLLM call consumers would make.
    # Call via our local provider and shape into a Router-like payload:
    resp = await codeworld_provider.acomplete(
        messages=messages,
        metrics=CODEWORLD_METRICS,
        iterations=MAX_ITERATIONS,
        allowed_languages=ALLOWED_LANGUAGES,
        request_timeout=MAX_SECONDS,
        temperature=TEMPERATURE,
        seed=SEED,
    )
    payload = resp
    print(json.dumps({"example_response": payload}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
