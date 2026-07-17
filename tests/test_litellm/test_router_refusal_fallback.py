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


def test_get_refusal_fallback_patterns_direct(monkeypatch):
    from litellm.router import _get_refusal_fallback_patterns

    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    assert _get_refusal_fallback_patterns() == []

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["a", "b"]))
    assert _get_refusal_fallback_patterns() == ["a", "b"]

    # Non-JSON value is tolerated as a single literal pattern, not raised.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", "not json[")
    assert _get_refusal_fallback_patterns() == ["not json["]

    # A JSON object (not an array) is also coerced to a single literal pattern.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps({"a": 1}))
    assert _get_refusal_fallback_patterns() == [json.dumps({"a": 1})]

    # Blank/non-string entries in an otherwise valid array are dropped.
    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["ok", "", 5, None]))
    assert _get_refusal_fallback_patterns() == ["ok"]


def test_text_matches_refusal_patterns_direct():
    from litellm.router import _text_matches_refusal_patterns

    assert _text_matches_refusal_patterns("I cannot assist", ["cannot assist"]) is True
    assert _text_matches_refusal_patterns("all good here", ["cannot assist"]) is False
    # A malformed regex among the patterns is skipped, not raised.
    assert (
        _text_matches_refusal_patterns("I cannot assist", ["[", "cannot assist"])
        is True
    )
    assert _text_matches_refusal_patterns("I cannot assist", ["["]) is False


def test_refusal_stream_hold_chars_direct(monkeypatch):
    from litellm.router import _refusal_stream_hold_chars

    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", raising=False)
    assert _refusal_stream_hold_chars() == 400

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "50")
    assert _refusal_stream_hold_chars() == 50

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "0")
    assert _refusal_stream_hold_chars() == 0

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "-5")
    assert _refusal_stream_hold_chars() == 0

    monkeypatch.setenv("LITELLM_REFUSAL_FALLBACK_STREAM_HOLD_CHARS", "not-a-number")
    assert _refusal_stream_hold_chars() == 400


def test_stream_hold_release_direct():
    """_release() drains and clears the held buffer and disarms the hold."""
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=1000, model="m")
    hold.process(_stream_chunk("hello"))
    hold.process(_stream_chunk(" world"))
    assert len(hold._held) == 2

    released = hold._release()
    assert len(released) == 2
    assert hold.active is False
    assert hold._held == []

    # Calling again returns nothing further: the buffer was already drained.
    assert hold._release() == []


def test_stream_hold_delta_state_direct():
    """_delta_state() extracts (matchable text, reasoning length, saw tool call)
    from a single stream item for every delta shape it must handle."""
    from litellm.router import _RefusalStreamHold
    from litellm.types.utils import ModelResponseStream

    assert _RefusalStreamHold._delta_state(_stream_chunk("hello")) == ("hello", 0, False)
    assert _RefusalStreamHold._delta_state(
        _stream_chunk(refusal="I cannot help")
    ) == ("I cannot help", 0, False)
    assert _RefusalStreamHold._delta_state(
        _stream_chunk(reasoning_content="thinking...")
    ) == ("", len("thinking..."), False)
    tool_call = {"id": "t1", "type": "function", "function": {"name": "f", "arguments": ""}}
    text, reasoning_len, saw_tool_call = _RefusalStreamHold._delta_state(
        _stream_chunk(None, tool_calls=[tool_call])
    )
    assert saw_tool_call is True
    # Empty choices (e.g. a trailing usage-only chunk) must not raise.
    assert _RefusalStreamHold._delta_state(ModelResponseStream(choices=[])) == ("", 0, False)


def test_stream_hold_raise_refusal_direct():
    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import _RefusalStreamHold

    hold = _RefusalStreamHold(patterns=["cannot assist"], hold_chars=400, model="my-model")
    with pytest.raises(MidStreamFallbackError) as exc:
        hold._raise_refusal()
    assert exc.value.model == "my-model"
    assert exc.value.is_pre_first_chunk is True
    assert isinstance(exc.value.original_exception, litellm.ContentPolicyViolationError)


# ---------------------------------------------------------------------------
# Responses API streaming: the hold must also cover /v1/responses streams,
# where Azure expresses a content-filter block as a terminal
# `response.incomplete` EVENT (HTTP 200), never an exception.
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _ev(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _incomplete_event(reason="content_filter"):
    return _ev(
        "response.incomplete",
        response=SimpleNamespace(
            incomplete_details=SimpleNamespace(reason=reason), usage=None
        ),
    )


def _responses_hold(hold_chars=400):
    from litellm.router import _ResponsesRefusalStreamHold

    return _ResponsesRefusalStreamHold(
        patterns=["cannot assist"], hold_chars=hold_chars, model="gpt-5.5"
    )


def test_responses_hold_raises_on_content_filter_terminal():
    """Azure jailbreak filter: 200 stream ending response.incomplete with
    reason=content_filter must be treated like a content-policy violation."""
    from litellm.exceptions import MidStreamFallbackError

    hold = _responses_hold()
    assert hold.process(_ev("response.created", response=SimpleNamespace())) == []
    assert hold.process(_ev("response.output_text.delta", delta="I'm")) == []
    with pytest.raises(MidStreamFallbackError) as exc:
        hold.process(_incomplete_event())
    assert isinstance(exc.value.original_exception, litellm.ContentPolicyViolationError)
    assert exc.value.is_pre_first_chunk is True

    # A non-filter incomplete (e.g. max_output_tokens) is NOT a block.
    hold = _responses_hold()
    assert hold.process(_incomplete_event(reason="max_output_tokens")) == []
    assert len(hold.flush()) == 1


def test_responses_hold_matches_refusal_text_delta():
    from litellm.exceptions import MidStreamFallbackError

    hold = _responses_hold()
    assert hold.process(_ev("response.output_text.delta", delta="I'm sorry, but I ")) == []
    with pytest.raises(MidStreamFallbackError):
        hold.process(_ev("response.output_text.delta", delta="cannot assist with that."))


def test_responses_hold_releases_on_reasoning_window_and_tool_calls():
    hold = _responses_hold(hold_chars=10)
    assert hold.process(_ev("response.reasoning_summary_text.delta", delta="thinking hard...")) != []
    assert hold.active is False

    hold = _responses_hold()
    released = hold.process(_ev("response.function_call_arguments.delta", delta='{"a"'))
    assert len(released) == 1
    assert hold.active is False


def test_responses_hold_inert_when_env_unset(monkeypatch):
    monkeypatch.delenv("LITELLM_REFUSAL_FALLBACK_PATTERNS", raising=False)
    router = _streaming_router("x")
    from litellm.router import _ResponsesRefusalStreamHold

    hold = router._refusal_stream_hold_for_call(
        {"model": "m"}, hold_cls=_ResponsesRefusalStreamHold
    )
    assert hold.active is False
    ev = _incomplete_event()
    assert hold.process(ev) == [ev]  # passthrough, no raise


@pytest.mark.asyncio
async def test_responses_streaming_content_filter_fails_over(monkeypatch):
    """Full _aresponses_streaming_iterator path: a held content-filter stream is
    replaced by the fallback stream; no source event leaks to the client."""
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["cannot assist"])
    )
    router = _streaming_router("unused")

    class FakeSource:
        completed_response = None

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield _ev("response.created", response=SimpleNamespace())
            yield _ev("response.output_text.delta", delta="I'm sorry, but I ")
            yield _incomplete_event()

        async def aclose(self):
            self.closed = True

    rescue_events = [
        _ev("response.output_text.delta", delta='{"correct":"yes"}'),
        _ev("response.completed", response=SimpleNamespace(usage=None)),
    ]

    class FakeFallback:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for e in rescue_events:
                yield e

        async def aclose(self):
            pass

    captured = {}

    async def fake_common_utils(**kwargs):
        captured["e"] = kwargs["e"]
        return FakeFallback()

    monkeypatch.setattr(
        router, "async_function_with_fallbacks_common_utils", fake_common_utils
    )

    wrapper = await router._aresponses_streaming_iterator(
        response=FakeSource(),
        initial_kwargs={"model": "m", "input": "judge this"},
    )
    seen = [item async for item in wrapper]
    assert seen == rescue_events  # only fallback events; nothing from the source
    assert isinstance(captured["e"], litellm.ContentPolicyViolationError)


@pytest.mark.asyncio
async def test_responses_streaming_normal_stream_passes_through(monkeypatch):
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["cannot assist"])
    )
    router = _streaming_router("unused")

    events = [
        _ev("response.created", response=SimpleNamespace()),
        _ev("response.output_text.delta", delta='{"extracted_final_answer":"0.186593"}'),
        _ev("response.completed", response=SimpleNamespace(usage=None)),
    ]

    class FakeSource:
        completed_response = None

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for e in events:
                yield e

        async def aclose(self):
            pass

    wrapper = await router._aresponses_streaming_iterator(
        response=FakeSource(), initial_kwargs={"model": "m", "input": "judge this"}
    )
    assert [item async for item in wrapper] == events


def test_responses_hold_blocked_terminal_with_dict_details_and_enum_type():
    """incomplete_details may arrive dict-shaped, and event.type may be the
    str-Enum member rather than a plain string — both must trigger."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    hold = _responses_hold()
    ev = _ev(
        ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
        response=SimpleNamespace(incomplete_details={"reason": "content_filter"}, usage=None),
    )
    with pytest.raises(MidStreamFallbackError):
        hold.process(ev)

    # Enum-typed text delta accumulates too.
    hold = _responses_hold()
    with pytest.raises(MidStreamFallbackError):
        hold.process(_ev(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, delta="I cannot assist."))


@pytest.mark.asyncio
async def test_responses_failover_closes_source_and_reports_fallback_terminal(monkeypatch):
    """The abandoned source is closed early, and completed_response reflects the
    FALLBACK's terminal (never the source's swallowed content-filter incomplete) —
    even when the fallback stream ends without a terminal event."""
    monkeypatch.setenv(
        "LITELLM_REFUSAL_FALLBACK_PATTERNS", json.dumps(["cannot assist"])
    )
    router = _streaming_router("unused")

    class FakeSource:
        completed_response = None
        closed = False

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield _ev("response.output_text.delta", delta="x")
            self.completed_response = _incomplete_event()  # latched like the real iterator
            yield self.completed_response

        async def aclose(self):
            self.closed = True

    class TerminalLessFallback:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield _ev("response.output_text.delta", delta="rescued")

        async def aclose(self):
            pass

    async def fake_common_utils(**kwargs):
        return TerminalLessFallback()

    monkeypatch.setattr(
        router, "async_function_with_fallbacks_common_utils", fake_common_utils
    )
    source = FakeSource()
    wrapper = await router._aresponses_streaming_iterator(
        response=source, initial_kwargs={"model": "m", "input": "judge this"}
    )
    seen = [item async for item in wrapper]
    assert [getattr(i, "delta", None) for i in seen] == ["rescued"]
    assert source.closed is True
    # The blocked source terminal must NOT be reported as the completed response.
    assert wrapper.completed_response is None


def test_is_blocked_terminal_direct():
    """Direct coverage of the _is_blocked_terminal hook on both classes."""
    from litellm.router import _RefusalStreamHold, _ResponsesRefusalStreamHold

    # Base hook: always False (chat has no blocked-terminal event shape).
    assert _RefusalStreamHold._is_blocked_terminal(_stream_chunk("x")) is False

    assert _ResponsesRefusalStreamHold._is_blocked_terminal(_incomplete_event()) is True
    assert (
        _ResponsesRefusalStreamHold._is_blocked_terminal(
            _incomplete_event(reason="max_output_tokens")
        )
        is False
    )
    assert (
        _ResponsesRefusalStreamHold._is_blocked_terminal(
            _ev("response.completed", response=SimpleNamespace(usage=None))
        )
        is False
    )


def test_fallback_dispatch_exception_unwraps_only_cpv():
    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import _fallback_dispatch_exception

    cpv = litellm.ContentPolicyViolationError(message="m", model="m", llm_provider="")
    wrapped = MidStreamFallbackError(
        message="m", model="m", llm_provider="", original_exception=cpv
    )
    assert _fallback_dispatch_exception(wrapped) is cpv

    plain = MidStreamFallbackError(
        message="m", model="m", llm_provider="", original_exception=ValueError("x")
    )
    assert _fallback_dispatch_exception(plain) is plain


def _mid_stream_error(original):
    from litellm.exceptions import MidStreamFallbackError

    return MidStreamFallbackError(
        message="m", model="m", llm_provider="", original_exception=original
    )


@pytest.mark.asyncio
async def test_maybe_abandon_refused_stream_source_closes_and_clears_latch():
    from litellm.router import _maybe_abandon_refused_stream_source

    cpv = litellm.ContentPolicyViolationError(message="m", model="m", llm_provider="")

    class Src:
        closed = False
        completed_response = "latched-blocked-terminal"

        async def aclose(self):
            self.closed = True

    src = Src()
    await _maybe_abandon_refused_stream_source(_mid_stream_error(cpv), src, "test")
    assert src.closed is True
    assert src.completed_response is None

    # Non-CPV mid-stream errors (429/5xx) leave the source untouched.
    other = Src()
    await _maybe_abandon_refused_stream_source(_mid_stream_error(ValueError("x")), other, "test")
    assert other.closed is False
    assert other.completed_response == "latched-blocked-terminal"

    # aclose raising must never propagate.
    class Angry:
        completed_response = "x"

        async def aclose(self):
            raise RuntimeError("boom")

    angry = Angry()
    await _maybe_abandon_refused_stream_source(_mid_stream_error(cpv), angry, "test")
    assert angry.completed_response is None


def test_prepare_responses_fallback_item_combines_partial_usage():
    from litellm.router import Router
    from litellm.types.llms.openai import ResponseAPIUsage, ResponseCompletedEvent

    usage = ResponseAPIUsage(input_tokens=3, output_tokens=4, total_tokens=7)
    # A plain event with no response/usage passes through untouched.
    plain = _ev("response.output_text.delta", delta="x")
    Router._prepare_responses_fallback_item(plain, None, usage)
    Router._prepare_responses_fallback_item(plain, None, None)
    assert plain.delta == "x"
