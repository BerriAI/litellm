"""
Gap tests for the stream wrappers of ``router_python_core.py`` (lines 2800-5600): what a mid-stream
fallback yields when the fallback answers without a stream, the usage on fallback chunks, and the
Responses stream's handling of an error that is not a mid-stream fallback error.

Each test asserts what the Python core does today; oddities are named in the test's docstring.
"""

from collections.abc import AsyncIterator, Callable
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.utils import ModelResponseStream, Usage
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    bad_request,
    completion_response,
    deployment,
    rate_limit,
)
from tests.unit.test_router.core.streaming_helpers import (
    MESSAGES,
    Event,
    chat_stream,
    drain_into,
    finish_chunk,
    responses_stream,
    text_chunk,
)

MakeCore = Callable[..., RouterCore]
GROUPS_AB: Final = [deployment("a", "a"), deployment("b", "b")]


def build(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    return make_core(**{"model_list": GROUPS_AB, "fallbacks": [{"a": ["b"]}], "num_retries": 0, **arguments})


def stream_with_usage(sync: bool = False) -> Callable[..., object]:
    """A bare chunk iterator (not a ``CustomStreamWrapper``, which would strip the usage) of one text
    chunk and a finish chunk that carries usage."""

    def answer(**_request: object) -> object:  # kwargs-ok: the host passes the request kwargs
        chunks: Final = [
            text_chunk("B"),
            finish_chunk(usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}),
        ]

        async def achunks() -> AsyncIterator[ModelResponseStream]:
            for chunk in chunks:
                yield chunk

        return iter(chunks) if sync else achunks()

    return answer


# --- a fallback that answers without a stream --------------------------------------------------


async def test_a_mid_stream_fallback_that_answers_without_a_stream_yields_one_none_item(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: the fallback's whole response is dropped and the stream yields ``None`` in its place."""
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", completion_response("not streamed", model="b"))
    core: Final = build(make_core)
    stream: Final = await core.acompletion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]
    seen: list[object] = []
    await drain_into(stream, seen)
    assert seen[-1] is None
    assert host.deployments_called() == ["a", "b"]


def test_a_sync_mid_stream_fallback_that_answers_without_a_stream_yields_the_response_fields(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: a ``ModelResponse`` is iterable (pydantic yields ``(field, value)`` pairs), so the sync
    wrapper streams the fallback response's fields instead of chunks."""
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True, sync=True))
    host.answer("openai/b", completion_response("not streamed", model="b"))
    core: Final = build(make_core)
    stream: Final = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]
    items: Final = list(stream)  # pyright: ignore[reportArgumentType]
    fields: Final = [item for item in items if isinstance(item, tuple)]
    assert ("model", "b") in fields
    assert {name for name, _ in fields} >= {"id", "choices", "usage"}
    assert host.deployments_called() == ["a", "b"]


# --- usage on fallback chunks ------------------------------------------------------------------


async def test_a_mid_stream_fallback_chunk_dict_usage_becomes_an_all_zero_usage(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: a fallback chunk's usage is rebuilt through ``combine_usage_objects``, which reads token
    counts as attributes, so a usage given as a plain dict comes out as an all-zero ``Usage``."""
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True))
    host.answer("openai/b", stream_with_usage())
    core: Final = build(make_core)
    stream: Final = await core.acompletion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]
    seen: list[object] = []
    await drain_into(stream, seen)
    usages: Final = [
        getattr(item, "usage")
        for item in seen
        if isinstance(item, ModelResponseStream) and getattr(item, "usage", None) is not None
    ]
    assert usages
    assert isinstance(usages[-1], Usage)
    assert (usages[-1].prompt_tokens, usages[-1].completion_tokens, usages[-1].total_tokens) == (0, 0, 0)


def test_a_sync_mid_stream_fallback_chunk_dict_usage_becomes_an_all_zero_usage(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: as in the async wrapper, a plain-dict usage comes out as an all-zero ``Usage``."""
    host.answer("openai/a", chat_stream(error=rate_limit(), leading_role_chunk=True, sync=True))
    host.answer("openai/b", stream_with_usage(sync=True))
    core: Final = build(make_core)
    stream: Final = core.completion(model="a", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]
    usages: Final = [
        getattr(item, "usage")
        for item in stream  # pyright: ignore[reportGeneralTypeIssues]
        if isinstance(item, ModelResponseStream) and getattr(item, "usage", None) is not None
    ]
    assert usages
    assert isinstance(usages[-1], Usage)
    assert (usages[-1].prompt_tokens, usages[-1].completion_tokens, usages[-1].total_tokens) == (0, 0, 0)


# --- Responses streams -------------------------------------------------------------------------


async def test_a_responses_error_that_is_not_a_mid_stream_error_flushes_held_events_then_raises(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.answer(
        "openai/a",
        responses_stream(Event("response.created"), Event("response.in_progress"), error=bad_request(message="plain")),
    )
    host.answer("openai/b", responses_stream(Event("response.created"), Event("response.completed")))
    core: Final = build(make_core)
    stream: Final = await core.aresponses(model="a", input="hi", stream=True)  # pyright: ignore[reportAttributeAccessIssue]
    seen: list[object] = []
    with pytest.raises(litellm.BadRequestError, match="plain"):
        await drain_into(stream, seen)
    assert [getattr(event, "type") for event in seen] == ["response.created", "response.in_progress"]
    assert host.deployments_called() == ["a"]


async def test_a_responses_stream_fallback_that_answers_without_a_stream_yields_that_response(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    from litellm.exceptions import MidStreamFallbackError

    cut: Final = MidStreamFallbackError(
        message="stream cut",
        model="a",
        llm_provider="anthropic",
        original_exception=rate_limit(message="stream cut"),
        generated_content="",
        is_pre_first_chunk=True,
    )
    plain: Final = completion_response("not streamed", model="b")
    host.answer("openai/a", responses_stream(Event("response.created"), error=cut))
    host.answer("openai/b", plain)
    core: Final = build(make_core)
    stream: Final = await core.aresponses(model="a", input="hi", stream=True)  # pyright: ignore[reportAttributeAccessIssue]
    seen: list[object] = []
    await drain_into(stream, seen)
    assert seen == [plain]
    assert host.deployments_called() == ["a", "b"]
