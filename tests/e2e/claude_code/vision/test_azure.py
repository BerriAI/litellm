"""vision x Azure.

Drive the real `claude` CLI against a running LiteLLM proxy that routes
to Azure, attach a small image as an inline base64 `image` content
block via the CLI's `--input-format stream-json` mode, and assert that
the upstream produces a non-empty reply. This proves the proxy
preserves Claude Code's multimodal content blocks end-to-end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/vision/test_azure.py
                       ^^^^^^      ^^^^^^^^^
                       feature_id  provider

Why stream-json input rather than `--image <path>`: the claude CLI
dropped `--image` in 2.x. Image attachments are now driven via either
the Files API (server-uploaded blobs referenced by file_id) or by
sending an Anthropic-shaped user message through stdin. We use the
latter because it requires no upstream pre-upload — the test stays
hermetic and the wire shape (an `image` content block) is exactly what
the proxy must preserve.
"""

from __future__ import annotations

import json
import pytest

from claude_code._env import require_proxy
from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)


AZURE_MODELS = [
    "claude-haiku-4-5-azure",
    "claude-sonnet-4-5-azure",
    "claude-opus-4-7-azure",
]

# Minimal 1x1 red PNG, base64-encoded. We embed it directly as the
# `image` content block's source — no temp file or Files API upload
# needed, the test stays hermetic.
RED_PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

VISION_PROMPT = (
    "What single color do you see in the attached image? Answer in one word."
)


def _build_stdin_input() -> str:
    """Build the newline-delimited JSON payload for `--input-format stream-json`.

    The CLI consumes a stream of `user` events whose `message.content` is
    a list of Anthropic content blocks. A single user event with one
    text block + one image block is enough to exercise the multimodal
    code path.
    """
    user_event = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": RED_PIXEL_PNG_B64,
                    },
                },
            ],
        },
    }
    return json.dumps(user_event) + "\n"


def test_vision_azure(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy with an image
    attached via stream-json input and assert a non-empty reply."""
    base_url, api_key = require_proxy(compat_result)

    outcomes = run_claude_models_parallel(
        models=AZURE_MODELS,
        # When using --input-format stream-json the CLI rejects a
        # positional prompt; the prompt + image come in via stdin.
        prompt=None,
        base_url=base_url,
        api_key=api_key,
        extra_args=["--input-format", "stream-json"],
        stdin_input=_build_stdin_input(),
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
