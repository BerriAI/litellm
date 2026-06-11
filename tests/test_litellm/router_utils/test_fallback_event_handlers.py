"""Tests for mid-stream fallback continuation message building.

Anthropic removed assistant prefill starting with Claude Sonnet 4.6 / Opus 4.6
(a prefilled assistant message returns a 400 error), so the mid-stream fallback
must use the documented user-message continuation pattern for those models:
https://platform.claude.com/docs/en/about-claude/models/migration-guide

Models whose registry entry says supports_assistant_prefill=true, has no value,
or is unknown keep the legacy prefill-resume behavior.
"""

import pytest

import litellm
from litellm.router_utils.fallback_event_handlers import (
    MID_STREAM_CONTINUATION_SYSTEM_PROMPT,
    build_mid_stream_continuation_messages,
)


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    """Pin capability lookups to the in-repo cost map so the tests exercise this
    PR's registry changes instead of the remote map."""
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()

MESSAGES = [{"role": "user", "content": "Plan my trip to Tokyo"}]
PARTIAL = "Here are the best flight options I found so"


def _build(model_group):
    return build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group=model_group,
    )


def _assert_legacy_prefill(result):
    assert len(result) == 3
    assert result[0] == MESSAGES[0]
    assert result[1] == {
        "role": "system",
        "content": MID_STREAM_CONTINUATION_SYSTEM_PROMPT,
    }
    assert result[2] == {
        "role": "assistant",
        "content": PARTIAL,
        "prefix": True,
    }


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-6",
        "vertex_ai/claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
)
def test_prefill_rejecting_models_get_user_continuation(model):
    """Claude Sonnet 4.6+/Opus 4.6+ reject assistant prefill with a 400 —
    the continuation must ride a user message and carry the partial text."""
    result = _build(model)
    assert len(result) == 2
    assert result[0] == MESSAGES[0]
    assert result[1]["role"] == "user"
    assert PARTIAL in result[1]["content"]
    assert "Continue from where you left off" in result[1]["content"]
    # No prefill anywhere — the conversation must end with a user message.
    assert all(m.get("prefix") is not True for m in result)


@pytest.mark.parametrize(
    "model",
    [
        "claude-3-5-sonnet-20241022",  # registry: supports_assistant_prefill=true
        "gpt-4",  # registry entry exists, capability field absent
        "definitely-not-a-real-model",  # unknown model → capability lookup fails
        None,  # no model group available
    ],
)
def test_other_models_keep_legacy_prefill_resume(model):
    """Anything not explicitly marked supports_assistant_prefill=false keeps the
    pre-existing prefill-resume behavior (back-compat)."""
    _assert_legacy_prefill(_build(model))


def test_prefill_rejecting_fallback_target_gets_user_continuation():
    """The same continuation messages go to every fallback target — a primary
    that supports prefill must still use the user continuation when any
    configured fallback target rejects it (e.g. claude-3-5 → claude-sonnet-4-6)."""
    result = build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group="claude-3-5-sonnet-20241022",
        fallbacks=[{"claude-3-5-sonnet-20241022": ["claude-sonnet-4-6"]}],
    )
    assert len(result) == 2
    assert result[1]["role"] == "user"
    assert PARTIAL in result[1]["content"]


def test_unrelated_fallback_groups_do_not_affect_prefill():
    """Fallback config for OTHER model groups must not flip this group's behavior."""
    result = build_mid_stream_continuation_messages(
        messages=MESSAGES,
        generated_content=PARTIAL,
        model_group="gpt-4",
        fallbacks=[{"some-other-model": ["claude-sonnet-4-6"]}],
    )
    _assert_legacy_prefill(result)
