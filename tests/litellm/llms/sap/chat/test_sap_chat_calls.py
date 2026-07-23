"""
Live blackbox tests for SAP GenAI Hub — reasoning and cache_control.

These tests make real API calls and are skipped when AICORE_SERVICE_KEY is
not set in the environment.  Run them manually:

    cd /path/to/litellm
    AICORE_SERVICE_KEY='...' pytest tests/litellm/llms/sap/chat/test_sap_chat_calls.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import litellm

pytestmark = pytest.mark.skipif(
    not os.environ.get("AICORE_SERVICE_KEY"),
    reason="AICORE_SERVICE_KEY not set — skipping live SAP tests",
)

_KEY = os.environ.get("AICORE_SERVICE_KEY", "")

# Tenant-available models (verified against /lm/deployments response)
_MODEL_REASONING = "o3"                         # o-series: supports reasoning_effort
_MODEL_THINKING = "anthropic--claude-4-sonnet"  # Claude 4: supports thinking
_MODEL_CACHE = "anthropic--claude-4-sonnet"     # Claude 4: supports cache_control


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _complete(model: str, messages: list, **kwargs) -> litellm.ModelResponse:
    return litellm.completion(
        model=f"sap/{model}",
        messages=messages,
        api_key=_KEY,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Reasoning — reasoning_effort
# ---------------------------------------------------------------------------


class TestReasoningEffort:
    """Verify reasoning_effort is accepted and a coherent reply is returned."""

    def test_o3_reasoning_effort_high(self):
        """o3 with reasoning_effort=high returns a non-empty reply."""
        response = _complete(
            _MODEL_REASONING,
            messages=[{"role": "user", "content": "What is 2 + 2? Reply with the number only."}],
            reasoning_effort="high",
        )
        assert response.choices[0].message.content.strip() in ("4", "4.")

    def test_o3_reasoning_effort_low(self):
        """o3 with reasoning_effort=low still returns a correct reply."""
        response = _complete(
            _MODEL_REASONING,
            messages=[{"role": "user", "content": "What is 3 + 3? Reply with the number only."}],
            reasoning_effort="low",
        )
        assert response.choices[0].message.content.strip() in ("6", "6.")


# ---------------------------------------------------------------------------
# Reasoning — thinking (Anthropic extended thinking)
# ---------------------------------------------------------------------------


class TestThinking:
    """Verify Anthropic thinking block is accepted and the call succeeds."""

    def test_claude_thinking_enabled(self):
        """Claude 4 with thinking=enabled completes without error."""
        response = _complete(
            _MODEL_THINKING,
            messages=[{"role": "user", "content": "What is 4 + 4? Reply with the number only."}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        choice = response.choices[0]
        assert choice.message.content.strip() in ("8", "8.")
        # reasoning_content is optional; when absent the attribute is not set.
        # The call completing without error is the primary assertion.


# ---------------------------------------------------------------------------
# cache_control
# ---------------------------------------------------------------------------


class TestCacheControlLive:
    """Verify cache_control content parts are forwarded and the call succeeds."""

    def test_cache_control_ephemeral_accepted(self):
        """Anthropic model accepts a message with cache_control ephemeral parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Reply with exactly the word PONG.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        response = _complete(_MODEL_CACHE, messages)
        assert "PONG" in response.choices[0].message.content

    def test_mixed_parts_cache_control_accepted(self):
        """Mixed content parts — one cached, one not — are forwarded cleanly."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant.",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "Reply with exactly the word PONG.",
                    },
                ],
            }
        ]
        response = _complete(_MODEL_CACHE, messages)
        assert "PONG" in response.choices[0].message.content
