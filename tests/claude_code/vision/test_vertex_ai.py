"""vision x Vertex AI.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
Claude requests to Anthropic's models on Google Cloud Vertex AI, attach
a small image via the CLI's `--image` flag, and assert that the
upstream produces a non-empty reply.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/vision/test_vertex_ai.py
                       ^^^^^^      ^^^^^^^^^
                       feature_id  provider
"""

from __future__ import annotations

import base64
import os

import pytest

from tests.claude_code.cli_driver import ClaudeCLIError, run_claude

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-6-vertex",
    "claude-opus-4-7-vertex",
]

RED_PIXEL_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.mark.parametrize("model", VERTEX_AI_MODELS)
def test_vision_vertex_ai(compat_result, model, tmp_path):
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

    try:
        result = run_claude(
            prompt="What single color do you see in the attached image? Answer in one word.",
            model=model,
            base_url=base_url,
            api_key=api_key,
            extra_args=["--image", str(image_path)],
        )
    except ClaudeCLIError as exc:
        compat_result.set({"status": "fail", "error": f"[{model}] {exc}"})
        pytest.fail(str(exc), pytrace=False)
        return

    if result.exit_code != 0:
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] claude CLI exited {result.exit_code}: {result.stderr.strip()}",
            }
        )
        pytest.fail(f"claude CLI exited {result.exit_code} for {model}", pytrace=False)
        return

    if not result.text.strip():
        compat_result.set(
            {
                "status": "fail",
                "error": f"[{model}] claude returned empty assistant text on a vision prompt",
            }
        )
        pytest.fail(f"empty reply for {model}", pytrace=False)
        return

    compat_result.set({"status": "pass"})
