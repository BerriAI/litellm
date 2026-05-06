"""vision x Azure (Microsoft Foundry).

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models hosted in Microsoft Foundry on
Azure, attach a small image via the CLI's `--image` flag, and assert
that the upstream produces a non-empty reply.

Foundry's Anthropic deployments accept image content blocks identically
to anthropic.com (text + image input on Haiku 4.5, Sonnet 4.6, and Opus
4.7); LiteLLM passes them through unchanged on the `azure_ai/claude-*`
route.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/vision/test_azure.py
                       ^^^^^^      ^^^^^
                       feature_id  provider
"""

from __future__ import annotations

import base64
import os

import pytest

from tests.claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-6-azure",
    "claude-opus-4-7-azure",
]

RED_PIXEL_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def test_vision_azure(compat_result, tmp_path):
    """Drive the `claude` CLI against the LiteLLM proxy with an image
    attached and assert a non-empty reply."""
    base_url = os.environ.get(PROXY_BASE_URL_ENV)
    api_key = os.environ.get(PROXY_API_KEY_ENV)
    if not base_url or not api_key:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"missing required env: set {PROXY_BASE_URL_ENV} and "
                    f"{PROXY_API_KEY_ENV} to point at a running LiteLLM proxy"
                ),
            }
        )
        pytest.fail(
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured", pytrace=False
        )

    image_path = tmp_path / "red_pixel.png"
    image_path.write_bytes(base64.b64decode(RED_PIXEL_PNG_B64))

    outcomes = run_claude_models_parallel(
        models=AZURE_MODELS,
        prompt="What single color do you see in the attached image? Answer in one word.",
        base_url=base_url,
        api_key=api_key,
        extra_args=["--image", str(image_path)],
    )

    failures = []
    for model in AZURE_MODELS:
        outcome = outcomes[model]
        if isinstance(outcome, ClaudeCLIError):
            error = f"[{model}] {outcome}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if outcome.exit_code != 0:
            error = f"[{model}] claude CLI failed: {failure_diagnostic(outcome)}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if not outcome.text.strip():
            error = f"[{model}] claude returned empty assistant text on a vision prompt"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
