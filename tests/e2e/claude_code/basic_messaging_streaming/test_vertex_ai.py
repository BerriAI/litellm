"""basic_messaging_streaming x Vertex AI.

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Claude requests to
Anthropic's models on Google Cloud Vertex AI, and report the outcome
via `compat_result`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_streaming/test_vertex_ai.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id                     provider
"""

from __future__ import annotations

from claude_code._basic_messaging import run_basic_messaging_cell
from claude_code.conftest import CompatResult

VERTEX_AI_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-6-vertex",
    "claude-opus-4-7-vertex",
]


def test_basic_messaging_streaming_vertex_ai(compat_result: CompatResult) -> None:
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty streamed reply (one row per Claude tier).
    """
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=VERTEX_AI_MODELS,
        prompt="Count from 1 to 5, one number per line.",
        verify_streaming=True,
    )
