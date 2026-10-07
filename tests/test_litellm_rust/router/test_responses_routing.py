"""The Rust router backend against the Python one on the Responses API: the same attempts, the
same fallbacks, and the same events reaching the caller."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from tests.test_litellm_rust.support.router_backends import BACKENDS, Backend, router_headers

pytestmark = pytest.mark.requires_rust_extension

GROUPS: Final = ("primary", "fb1", "fb2")


def _arguments(fallbacks: Sequence[Mapping[str, Sequence[str]]], num_retries: int = 1) -> Mapping[str, object]:
    return {
        "model_list": tuple(
            {
                "model_name": group,
                "litellm_params": {"model": f"openai/{group}-model", "api_key": "k", "mock_response": f"from {group}"},
            }
            for group in GROUPS
        ),
        "fallbacks": [dict(entry) for entry in fallbacks],
        "num_retries": num_retries,
    }


def _event(kind: str, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(type=kind, delta=text)


class _Stream(BaseResponsesAPIStreamingIterator):
    """A provider stream: `events`, then `failure` if there is one."""

    def __init__(self, model: str, events: Sequence[SimpleNamespace], failure: MidStreamFallbackError | None) -> None:
        self.model = model
        self.completed_response = None
        self._hidden_params = {}
        self._events: Final = iter(tuple(events))
        self._failure: Final = failure

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        event: Final = next(self._events, None)
        if event is not None:
            return event
        if self._failure is not None:
            raise self._failure
        raise StopAsyncIteration

    async def aclose(self) -> None:
        return None


def _mid_stream(model: str, generated: str) -> MidStreamFallbackError:
    return MidStreamFallbackError(
        message=f"provider 500 from {model}",
        model=model,
        llm_provider="openai",
        generated_content=generated,
        is_pre_first_chunk=not generated,
        original_exception=litellm.InternalServerError(
            message=f"provider 500 from {model}", model=model, llm_provider="openai"
        ),
    )


Script = Mapping[str, str]


def _provider(
    script: Script, calls: list[tuple[str, object]]
) -> Callable[..., object]:  # mutable-ok: records each call
    """`litellm.aresponses` answering each group by its script: `ok`, `raise`, `drop-before-output`
    (an announcement, then a mid-stream error) or `drop-after-output` (some text, then the error)."""

    async def aresponses(**kwargs: object) -> object:  # kwargs-ok: litellm.aresponses' surface
        metadata: Final = kwargs["litellm_metadata"]
        assert isinstance(metadata, Mapping)
        group: Final = str(metadata["model_group"])
        model: Final = str(kwargs["model"])
        calls.append((group, kwargs.get("input")))
        behavior: Final = script.get(group, "ok")
        if behavior == "raise":
            raise litellm.InternalServerError(message=f"500 from {group}", model=model, llm_provider="openai")
        if behavior == "drop-before-output":
            return _Stream(model, (_event("response.created"),), _mid_stream(model, ""))
        if behavior == "drop-after-output":
            return _Stream(
                model,
                (_event("response.created"), _event("response.output_text.delta", f"partial from {group}")),
                _mid_stream(model, f"partial from {group}"),
            )
        return _Stream(
            model,
            (
                _event("response.created"),
                _event("response.output_text.delta", f"from {group}"),
                _event("response.completed"),
            ),
            None,
        )

    return aresponses


Observed = tuple[tuple[str, ...], tuple[tuple[str, object], ...], Mapping[str, object]]


async def _stream(
    build: Callable[[Mapping[str, object], int], Backend], arguments: Mapping[str, object], script: Script
) -> Observed:
    """`build` runs under the patch: Python's router binds `litellm.aresponses` when it is built."""
    calls: Final[list[tuple[str, object]]] = []  # mutable-ok: one entry per provider call
    events: Final[list[str]] = []  # mutable-ok: the events in the order the caller got them
    response: object = None  # rebind-ok: set once the call returns
    with patch("litellm.aresponses", side_effect=_provider(script, calls)):
        backend: Final = build(arguments, 1)
        try:
            response = await backend.aresponses(model="primary", input="hi", stream=True)  # rebind-ok: see above
            async for event in response:  # pyright: ignore[reportGeneralTypeIssues,reportUnknownVariableType]  # an async stream
                events.append(f"{event.type}:{event.delta}")  # pyright: ignore[reportUnknownMemberType]  # the scripted events
        except litellm.InternalServerError as error:
            events.append(f"raised {error.message}")
    return tuple(events), tuple(calls), router_headers(response)


@pytest.mark.parametrize(
    ("fallbacks", "script", "expected_events", "expected_groups"),
    (
        pytest.param(
            [{"primary": ["fb1"]}],
            {"primary": "drop-before-output"},
            ("response.created:", "response.output_text.delta:from fb1", "response.completed:"),
            ("primary", "fb1"),
            id="drop-before-output-falls-back-without-retrying",
        ),
        pytest.param(
            [{"primary": ["fb1"]}],
            {"primary": "drop-after-output"},
            (
                "response.created:",
                "response.output_text.delta:partial from primary",
                "response.created:",
                "response.output_text.delta:from fb1",
                "response.completed:",
            ),
            ("primary", "fb1"),
            id="drop-after-output-continues-on-the-fallback",
        ),
        pytest.param(
            [{"primary": ["fb1"]}, {"fb1": ["fb2"]}],
            {"primary": "drop-after-output", "fb1": "drop-before-output"},
            (
                "response.created:",
                "response.output_text.delta:partial from primary",
                "response.created:",
                "response.output_text.delta:from fb2",
                "response.completed:",
            ),
            ("primary", "fb1", "fb2"),
            id="a-fallback-that-drops-uses-its-own-chain",
        ),
        pytest.param(
            [{"primary": ["fb1"]}],
            {"primary": "drop-after-output", "fb1": "raise"},
            (
                "response.created:",
                "response.output_text.delta:partial from primary",
                "raised litellm.InternalServerError: provider 500 from openai/primary-model",
            ),
            ("primary", "fb1", "fb1"),
            id="the-provider-error-surfaces-when-the-fallback-fails",
        ),
    ),
)
async def test_a_dropped_responses_stream_falls_back_the_same_way_on_both_backends(
    fallbacks: Sequence[Mapping[str, Sequence[str]]],
    script: Script,
    expected_events: tuple[str, ...],
    expected_groups: tuple[str, ...],
) -> None:
    observed: Final = [await _stream(build, _arguments(fallbacks), script) for build in BACKENDS]

    assert observed[1][:2] == observed[0][:2]
    events, calls, _ = observed[1]
    assert events == expected_events
    assert tuple(group for group, _ in calls) == expected_groups


async def test_a_fallback_after_output_reports_the_fallback_in_both_backends_headers() -> None:
    observed: Final = [
        await _stream(build, _arguments([{"primary": ["fb1"]}]), {"primary": "drop-after-output"}) for build in BACKENDS
    ]

    assert observed[1][2] == observed[0][2]
    assert observed[1][2] == {
        "x-litellm-model-group": "fb1",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
    }


async def test_headers_name_the_group_that_served_after_a_fallback_falls_back_again() -> None:
    """Python's Responses wrapper adopts its fallback's headers before reading it, so when that
    fallback falls back again it still names the first one; its chat wrapper re-adopts them once the
    stream yields. The Rust backend names the group that served, as chat does."""
    _, _, headers = await _stream(
        BACKENDS[1],
        _arguments([{"primary": ["fb1"]}, {"fb1": ["fb2"]}]),
        {"primary": "drop-after-output", "fb1": "drop-before-output"},
    )

    assert headers == {
        "x-litellm-model-group": "fb2",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 2,
    }


async def test_a_fallback_after_output_continues_from_the_generated_text() -> None:
    calls: Final[list[tuple[str, object]]] = []  # mutable-ok: one entry per provider call
    backend: Final = BACKENDS[1](_arguments([{"primary": ["fb1"]}]), 1)
    with patch("litellm.aresponses", side_effect=_provider({"primary": "drop-after-output"}, calls)):
        response = await backend.aresponses(model="primary", input="hi", stream=True)
        _ = [event async for event in response]  # pyright: ignore[reportGeneralTypeIssues,reportUnknownVariableType]  # an async stream

    fallback_input: Final = calls[1][1]
    assert isinstance(fallback_input, list)
    assert fallback_input[-1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "partial from primary"}],
    }


@pytest.mark.parametrize("build", BACKENDS)
async def test_a_failing_group_falls_back_to_its_mock_response(
    build: Callable[[Mapping[str, object], int], Backend],
) -> None:
    backend: Final = build(_arguments([{"primary": ["fb1"]}], num_retries=0), 1)

    response: Final = await backend.aresponses(model="primary", input="hi", mock_testing_fallbacks=True)

    assert response.output[0].content[0].text == "from fb1"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]  # ResponsesAPIResponse
    assert router_headers(response) == {
        "x-litellm-model-group": "fb1",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
    }


@pytest.mark.parametrize("build", BACKENDS)
async def test_anthropic_messages_fall_back_to_the_fallback_groups_mock_response(
    build: Callable[[Mapping[str, object], int], Backend],
) -> None:
    backend: Final = build(_arguments([{"primary": ["fb1"]}], num_retries=0), 1)

    response: Final = await backend.aanthropic_messages(
        model="primary", messages=[{"role": "user", "content": "hi"}], max_tokens=10, mock_testing_fallbacks=True
    )

    assert isinstance(response, Mapping)
    assert response["content"][0]["text"] == "from fb1"
