"""Tests for the refusal -> content-policy-fallback bridge (rayward fork patch).

A model-level refusal is an HTTP 200 completion with finish_reason=stop and a
polite decline (e.g. "I'm sorry, but I cannot assist with that request."). Azure's
content filter never expresses that as finish_reason=content_filter, so without
this bridge such a refusal cannot fail over. When LITELLM_REFUSAL_FALLBACK_PATTERNS
is set (a JSON array of regexes), `_should_raise_content_policy_error` treats a
matching refusal like a content-policy violation so content_policy_fallbacks can
route it elsewhere.

The feature is OFF unless the env var is set — these tests assert the vanilla path
is unchanged when it is absent.
"""

import json

import litellm
from litellm.router import _completion_matches_refusal_patterns
from litellm.types.utils import Choices, Message, ModelResponse


def _response(content, finish_reason="stop"):
    return ModelResponse(
        choices=[Choices(finish_reason=finish_reason, message=Message(content=content))]
    )


def test_refusal_matcher_inert_when_env_unset(monkeypatch):
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    assert (
        _completion_matches_refusal_patterns(
            _response("I'm sorry, but I cannot assist with that request.")
        )
        is False
    )


def test_refusal_matcher_matches_refusal_ignores_normal(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS",
        json.dumps([r"i'?m sorry.*cannot assist", "i cannot help with"]),
    )
    assert _completion_matches_refusal_patterns(
        _response("I'm sorry, but I cannot assist with that request.")
    )
    assert not _completion_matches_refusal_patterns(
        _response('{"extracted_final_answer":"0.186593","correct":"yes"}')
    )
    assert not _completion_matches_refusal_patterns(_response(None))


def test_refusal_matcher_preserves_commas_inside_regex(monkeypatch):
    """A JSON array means a pattern may contain commas — the comma is a literal in
    the regex, not a separator, so a partial phrase must NOT trigger it."""
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["I'm sorry, but I cannot"])
    )
    assert _completion_matches_refusal_patterns(
        _response("I'm sorry, but I cannot assist with that request.")
    )
    assert not _completion_matches_refusal_patterns(_response("I'm sorry"))


def test_refusal_matcher_malformed_regex_never_raises(monkeypatch):
    """A malformed pattern must be skipped, never escape as a request failure; a
    valid sibling pattern still matches."""
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["[", "cannot assist"])
    )
    assert _completion_matches_refusal_patterns(
        _response("I'm sorry, but I cannot assist with that request.")
    )
    # A non-JSON value is tolerated as a single pattern rather than raising.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", "cannot assist")
    assert _completion_matches_refusal_patterns(
        _response("...I cannot assist...")
    )


def _router():
    return litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x"}},
            {"model_name": "m-openai", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "y"}},
        ],
        content_policy_fallbacks=[{"m": ["m-openai"]}],
    )


def test_should_raise_on_refusal_only_when_enabled(monkeypatch):
    router = _router()
    refusal = _response("I'm sorry, but I cannot assist with that request.")

    # Vanilla: a stop-refusal is NOT a content-policy error.
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    assert router._should_raise_content_policy_error("m", refusal, {}) is False

    # Enabled: the matching refusal now raises so content_policy_fallbacks fires.
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps([r"i'?m sorry.*cannot assist"])
    )
    assert router._should_raise_content_policy_error("m", refusal, {}) is True

    # A normal answer never trips it, even when enabled.
    assert (
        router._should_raise_content_policy_error("m", _response("FINAL=0.186593"), {})
        is False
    )


def test_content_filter_finish_reason_still_raises_without_env(monkeypatch):
    """The original content_filter path must keep working with the env unset."""
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    router = _router()
    filtered = _response("", finish_reason="content_filter")
    assert router._should_raise_content_policy_error("m", filtered, {}) is True
