"""
Live / blackbox tests for the SAP GenAI Hub orchestration provider.

These tests make real network calls to SAP AI Core.  They are skipped
automatically when AICORE_SERVICE_KEY is not set, so CI never runs them.

Run locally:
    export AICORE_SERVICE_KEY='{ ... json ... }'
    pytest tests/litellm/llms/sap/chat/test_sap_chat_calls.py -v

All tests use the default SAP model (gpt-4o via the orchestration service)
unless a specific model is needed for the feature under test.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AICORE_SERVICE_KEY"),
    reason="Live SAP credentials not available (AICORE_SERVICE_KEY not set)",
)

_KEY = os.environ.get("AICORE_SERVICE_KEY", "")
_SAP_ANTHROPIC_37 = "sap/anthropic--claude-3-7-sonnet-20250219"
_SAP_ANTHROPIC_35 = "sap/anthropic--claude-3-5-sonnet"
_SAP_O4 = "sap/o4-mini"
_PING = [{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_live_response(response) -> None:
    assert response is not None
    assert response.choices, "response.choices is empty"
    content = response.choices[0].message.content or ""
    assert content.strip(), "response content is empty"


# ---------------------------------------------------------------------------
# reasoning_effort / thinking params
# ---------------------------------------------------------------------------


class TestReasoningParams:
    """Verify reasoning_effort and thinking params reach SAP models."""

    def test_reasoning_effort_anthropic(self):
        """Anthropic claude-3-7 accepts reasoning_effort."""
        import litellm

        response = litellm.completion(
            model=_SAP_ANTHROPIC_37,
            messages=_PING,
            api_key=_KEY,
            reasoning_effort="low",
        )
        _assert_live_response(response)

    def test_thinking_anthropic(self):
        """Anthropic claude-3-7 accepts thinking dict."""
        import litellm

        response = litellm.completion(
            model=_SAP_ANTHROPIC_37,
            messages=_PING,
            api_key=_KEY,
            thinking={"type": "enabled", "budget_tokens": 500},
            max_tokens=1024,
        )
        _assert_live_response(response)

    def test_reasoning_effort_o_series(self):
        """OpenAI o4-mini on SAP accepts reasoning_effort."""
        import litellm

        response = litellm.completion(
            model=_SAP_O4,
            messages=_PING,
            api_key=_KEY,
            reasoning_effort="low",
        )
        _assert_live_response(response)


# ---------------------------------------------------------------------------
# cache_control passthrough
# ---------------------------------------------------------------------------


class TestCacheControl:
    """Verify cache_control on message content parts reaches SAP/Anthropic."""

    def _build_cached_messages(self, system_text: str):
        """Messages with cache_control on the system prompt content part."""
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "Reply PONG."},
        ]

    def test_cache_control_request_succeeds(self):
        """A request with cache_control metadata completes without error."""
        import litellm

        # System prompt well above Anthropic's 1024-token minimum for caching.
        system = "You are a helpful assistant. " * 200
        response = litellm.completion(
            model=_SAP_ANTHROPIC_35,
            messages=self._build_cached_messages(system),
            api_key=_KEY,
            max_tokens=64,
        )
        _assert_live_response(response)

    def test_cache_control_second_call_shows_cache_read(self):
        """Second identical call should read from Anthropic prompt cache.

        cache_read_input_tokens > 0 on the second call proves the cache_control
        metadata reached Anthropic and was honoured.
        """
        import litellm

        system = "You are a helpful assistant. " * 200
        msgs = self._build_cached_messages(system)

        # First call: writes the cache
        litellm.completion(
            model=_SAP_ANTHROPIC_35, messages=msgs, api_key=_KEY, max_tokens=64
        )
        # Second call: should read from cache
        r2 = litellm.completion(
            model=_SAP_ANTHROPIC_35, messages=msgs, api_key=_KEY, max_tokens=64
        )
        _assert_live_response(r2)
        cache_read = getattr(r2.usage, "cache_read_input_tokens", 0) or 0
        assert cache_read > 0, (
            "Expected cache_read_input_tokens > 0 on second call "
            f"(got {cache_read}); cache_control may not have reached Anthropic"
        )
