"""
Cassette-replayed Anthropic completion tests.

These tests exercise the same end-to-end ``litellm.completion`` code paths as
``test_anthropic_completion.py`` but replay HTTP traffic from cassettes under
``cassettes/test_anthropic_completion_vcr/`` instead of calling
``api.anthropic.com``. CI runs them with no API key and zero cost.

Add a new test by writing it normally and decorating with ``@pytest.mark.vcr``.
The cassette path is resolved automatically from the test module + test name
by ``pytest-recording`` (see ``conftest.py``).

To re-record every marked test in one sweep::

    ANTHROPIC_API_KEY=sk-ant-... \\
        uv run pytest tests/llm_translation -m vcr --record-mode=once

See ``tests/llm_translation/cassettes/README.md`` for the full workflow.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm  # noqa: E402

# A non-secret placeholder API key. The vcr_config fixture in conftest.py
# filters Authorization / x-api-key headers from cassettes, so this value
# never lands on disk; it only stops the SDK from raising when
# ``ANTHROPIC_API_KEY`` is unset (the common CI case).
PLACEHOLDER_ANTHROPIC_API_KEY = "sk-ant-vcr-placeholder"


@pytest.fixture(autouse=True)
def _placeholder_anthropic_key(monkeypatch):
    """Ensure an API key is set so replay works offline.

    If a real key is present (e.g. when re-recording with
    ``--record-mode=once``), we leave it untouched.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", PLACEHOLDER_ANTHROPIC_API_KEY)


@pytest.mark.vcr
def test_anthropic_basic_completion_replay():
    """Smoke-test that a vanilla Anthropic completion replays from a cassette.

    This is the canonical example for the cassette-based testing pattern: no
    API key required at runtime, deterministic output, and the full LiteLLM
    transformation pipeline (request shaping + response parsing) runs against
    a real-shape Anthropic payload.
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


@pytest.mark.vcr
def test_anthropic_streaming_completion_replay():
    """Replay a streaming Anthropic completion from a cassette.

    Exercises the SSE chunk parser and the public streaming surface. The
    underlying cassette captures every ``content_block_delta`` event Anthropic
    emits, so any regression in the streaming transformation will surface
    here.
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
