"""
VCR-backed Anthropic completion tests.

These tests exercise the same end-to-end ``litellm.completion`` code paths
as ``test_anthropic_completion.py`` but replay HTTP traffic from cassettes
under ``cassettes/`` instead of calling ``api.anthropic.com``. CI can run
them with no API key and zero cost.

To re-record after a deliberate change to request shape (or to refresh
against the live API), set ``LITELLM_VCR_RECORD_MODE=once`` and provide a
real ``ANTHROPIC_API_KEY``::

    LITELLM_VCR_RECORD_MODE=once \\
        ANTHROPIC_API_KEY=sk-ant-... \\
        uv run pytest tests/llm_translation/test_anthropic_completion_vcr.py -v

See ``tests/llm_translation/vcr_config.py`` and ``tests/llm_translation/cassettes/README.md``
for the full workflow.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.dirname(__file__))

import litellm  # noqa: E402

from vcr_config import litellm_vcr  # noqa: E402


# A non-secret placeholder API key. We never want a real key written to a
# cassette, and ``vcr_config`` filters Authorization / x-api-key headers
# anyway. Using a deterministic placeholder also stops the SDK from raising
# when ``ANTHROPIC_API_KEY`` is unset (the common CI case).
PLACEHOLDER_ANTHROPIC_API_KEY = "sk-ant-vcr-placeholder"


@pytest.fixture(autouse=True)
def _placeholder_anthropic_key(monkeypatch):
    """Provide a placeholder key when none is set so replay works offline.

    If a real key is present in the environment (e.g. when re-recording),
    we leave it untouched.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", PLACEHOLDER_ANTHROPIC_API_KEY)


@litellm_vcr.use_cassette("anthropic_basic_completion.yaml")
def test_anthropic_basic_completion_replay():
    """Smoke-test that a vanilla Anthropic completion replays from a cassette.

    This is the canonical example for the cassette-based testing pattern:
    no API key required at runtime, deterministic output, and the full
    LiteLLM transformation pipeline (request shaping + response parsing)
    runs against a real-shape Anthropic payload.
    """
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert response is not None
    assert response.choices[0].message.content == ("Hello! How can I help you today?")
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 11
    # Anthropic sets stop_reason="end_turn" → litellm normalises to "stop"
    assert response.choices[0].finish_reason == "stop"


@litellm_vcr.use_cassette("anthropic_streaming_completion.yaml")
def test_anthropic_streaming_completion_replay():
    """Replay a streaming Anthropic completion from a cassette.

    Exercises the SSE chunk parser and the public streaming surface. The
    underlying cassette captures every ``content_block_delta`` event Anthropic
    emits, so any regression in the streaming transformation will surface here.
    """
    stream = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    collected_text = ""
    finish_reason = None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            collected_text += delta.content
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    assert collected_text == "Hello from LiteLLM!"
    assert finish_reason == "stop"
