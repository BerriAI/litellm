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

import pytest

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


# ---------------------------------------------------------------------------
# Streaming: the refusal bridge must also cover stream=True. The head of the
# stream is held back until a refusal is ruled out, so a streamed refusal is
# replaced by the fallback stream and the client never sees the refusal text.
# ---------------------------------------------------------------------------

REFUSAL = "I'm sorry, but I cannot assist with that request."


def _streaming_router(primary_response, fallback_response="FINAL=42"):
    return litellm.Router(
        model_list=[
            {
                "model_name": "m",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": primary_response,
                },
            },
            {
                "model_name": "m-openai",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "y",
                    "mock_response": fallback_response,
                },
            },
        ],
        content_policy_fallbacks=[{"m": ["m-openai"]}],
    )


async def _collect_stream_text(stream) -> str:
    text = ""
    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        content = getattr(getattr(choices[0], "delta", None), "content", None)
        if isinstance(content, str):
            text += content
    return text


@pytest.mark.asyncio
async def test_streaming_refusal_fails_over_when_enabled(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps([r"i'?m sorry.*cannot assist"])
    )
    router = _streaming_router(REFUSAL)
    stream = await router.acompletion(
        model="m", messages=[{"role": "user", "content": "grade this"}], stream=True
    )
    text = await _collect_stream_text(stream)
    assert "cannot assist" not in text.lower()
    assert text == "FINAL=42"


@pytest.mark.asyncio
async def test_streaming_refusal_passthrough_when_env_unset(monkeypatch):
    """Vanilla behavior with the env unset: the refusal streams through."""
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    router = _streaming_router(REFUSAL)
    stream = await router.acompletion(
        model="m", messages=[{"role": "user", "content": "grade this"}], stream=True
    )
    assert await _collect_stream_text(stream) == REFUSAL


@pytest.mark.asyncio
async def test_streaming_normal_answer_unaffected_when_enabled(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps([r"i'?m sorry.*cannot assist"])
    )
    router = _streaming_router("The verdict is correct.")
    stream = await router.acompletion(
        model="m", messages=[{"role": "user", "content": "grade this"}], stream=True
    )
    assert await _collect_stream_text(stream) == "The verdict is correct."


def test_stream_hold_disabled_without_content_policy_fallback(monkeypatch):
    """Holding delays TTFB, so it must only arm for model groups that actually
    have a content_policy_fallbacks entry (and only when patterns are set)."""
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["cannot assist"])
    )
    router = _streaming_router(REFUSAL)
    armed = router._refusal_stream_hold_for_call({"model": "m"})
    assert armed.active is True
    unarmed = router._refusal_stream_hold_for_call({"model": "no-fallback-group"})
    assert unarmed.active is False

    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    assert router._refusal_stream_hold_for_call({"model": "m"}).active is False


def _stream_chunk(content=None, tool_calls=None, reasoning_content=None, refusal=None):
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    delta = Delta(content=content, tool_calls=tool_calls)
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    if refusal is not None:
        delta.refusal = refusal
    return ModelResponseStream(choices=[StreamingChoices(delta=delta)])


def test_stream_hold_releases_after_limit_and_on_tool_calls(monkeypatch):
    """A long non-refusal stream is released once the hold limit is hit, and a
    tool-call delta releases immediately (refusals are plain text)."""
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=10, model="m")
    assert hold.process(_stream_chunk("hello ")) == []
    released = hold.process(_stream_chunk("world!!!"))
    assert len(released) == 2  # both held chunks flushed once limit is reached
    assert hold.process(_stream_chunk("more")) != []  # passthrough afterwards

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=1000, model="m")
    assert hold.process(_stream_chunk("thinking")) == []
    tool_call = {"id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}
    assert len(hold.process(_stream_chunk(None, tool_calls=[tool_call]))) == 2
    assert hold.flush() == []

    # Empty-choices chunks (e.g. a trailing usage chunk) are held, not crashed on.
    from litellm.types.utils import ModelResponseStream

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=10, model="m")
    assert hold.process(ModelResponseStream(choices=[])) == []
    assert len(hold.flush()) == 1


def test_stream_hold_reasoning_deltas_advance_the_window():
    """Reasoning deltas can't match the visible refusal but must still advance
    the hold window — otherwise a reasoning model's whole thinking phase (and
    response) would be buffered to end-of-stream."""
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=10, model="m")
    assert hold.process(_stream_chunk(reasoning_content="let me think about it")) != []
    assert hold.active is False


def test_stream_hold_matches_openai_refusal_delta():
    """OpenAI structured-output refusals arrive in delta.refusal, not content."""
    import pytest as _pytest

    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=400, model="m")
    with _pytest.raises(MidStreamFallbackError) as exc:
        hold.process(_stream_chunk(refusal="I cannot assist with that."))
    assert isinstance(exc.value.original_exception, litellm.ContentPolicyViolationError)
    assert exc.value.is_pre_first_chunk is True


def test_stream_hold_disarmed_for_multi_choice_requests(monkeypatch):
    """n>1 interleaves choice deltas; a single buffer can't match reliably, so
    the hold must stay off and the stream must pass through vanilla."""
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["cannot assist"])
    )
    router = _streaming_router(REFUSAL)
    assert router._refusal_stream_hold_for_call({"model": "m", "n": 2}).active is False
    assert router._refusal_stream_hold_for_call({"model": "m", "n": 1}).active is True
    assert router._refusal_stream_hold_for_call({"model": "m"}).active is True
