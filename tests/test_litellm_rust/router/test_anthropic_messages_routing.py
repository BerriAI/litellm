"""The Rust router backend against the Python one on streamed Anthropic messages: the same
same-group retries, the same fallbacks, and the same frames reaching the caller."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from tests.test_litellm_rust.support.router_backends import BACKENDS, Backend, router_headers

pytestmark = pytest.mark.requires_rust_extension

GROUPS: Final = ("primary", "fb1")


def _arguments(fallbacks: Sequence[Mapping[str, Sequence[str]]], num_retries: int) -> Mapping[str, object]:
    return {
        "model_list": tuple(
            {"model_name": group, "litellm_params": {"model": f"anthropic/{group}-model", "api_key": "k"}}
            for group in GROUPS
        ),
        "fallbacks": [dict(entry) for entry in fallbacks],
        "num_retries": num_retries,
    }


def _frame(event: str, data: Mapping[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


START: Final = _frame("message_start", {"type": "message_start", "message": {"id": "m", "content": []}})
PING: Final = _frame("ping", {"type": "ping"})
OVERLOADED: Final = _frame("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
INVALID: Final = _frame("error", {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})


def _delta(text: str) -> bytes:
    return _frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    )


Script = Mapping[str, Sequence[Sequence[bytes] | None]]


def _provider(script: Script, calls: list[str]) -> Callable[..., object]:  # mutable-ok: records each call
    """`litellm.anthropic_messages` answering each group's calls, in order, with that call's frames
    (`None` raises before the stream opens); calls past the script answer with a content delta
    naming the group."""

    async def anthropic_messages(
        **kwargs: object,
    ) -> AsyncIterator[bytes]:  # kwargs-ok: litellm.anthropic_messages' surface
        metadata: Final = kwargs["litellm_metadata"]
        assert isinstance(metadata, Mapping)
        group: Final = str(metadata["model_group"])
        index: Final = sum(1 for called in calls if called == group)
        calls.append(group)
        answers: Final = script.get(group, ())
        frames: Final = answers[index] if index < len(answers) else (START, _delta(f"from {group}"))
        if frames is None:
            raise litellm.InternalServerError(message=f"500 from {group}", model=group, llm_provider="anthropic")

        async def stream() -> AsyncIterator[bytes]:
            for frame in frames:
                yield frame

        return stream()

    return anthropic_messages


Observed = tuple[tuple[bytes, ...], tuple[str, ...], Mapping[str, object]]


async def _stream(
    build: Callable[[Mapping[str, object], int], Backend], arguments: Mapping[str, object], script: Script
) -> Observed:
    """`build` runs under the patch: Python's router binds `litellm.anthropic_messages` when it is built."""
    calls: Final[list[str]] = []  # mutable-ok: one entry per provider call
    with patch("litellm.anthropic_messages", side_effect=_provider(script, calls)):
        backend: Final = build(arguments, 1)
        response: Final = await backend.aanthropic_messages(
            model="primary", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=True
        )
        frames: Final = tuple([frame async for frame in response])  # pyright: ignore[reportGeneralTypeIssues,reportUnknownVariableType]  # a byte stream
    return frames, tuple(calls), router_headers(response)


@pytest.mark.parametrize(
    ("fallbacks", "num_retries", "script", "expected_frames", "expected_calls"),
    (
        pytest.param(
            [{"primary": ["fb1"]}],
            1,
            {"primary": ((START, OVERLOADED), (START, OVERLOADED))},
            (START, _delta("from fb1")),
            ("primary", "primary", "fb1"),
            id="an-overloaded-group-retries-then-falls-back",
        ),
        pytest.param(
            [],
            2,
            {"primary": ((START, OVERLOADED),)},
            (START, _delta("from primary")),
            ("primary", "primary"),
            id="a-retry-that-opens-serves-the-stream",
        ),
        pytest.param(
            [],
            0,
            {"primary": ((START, OVERLOADED),)},
            (START, OVERLOADED),
            ("primary",),
            id="an-unrecoverable-stream-forwards-the-error-frame",
        ),
        pytest.param(
            [{"primary": ["fb1"]}],
            1,
            {"primary": ((START, INVALID),)},
            (START, INVALID),
            ("primary",),
            id="a-client-error-frame-is-not-retried",
        ),
        pytest.param(
            [],
            1,
            {"primary": ((PING, START, PING, _delta("hello")),)},
            (PING, PING, START, _delta("hello")),
            ("primary",),
            id="pings-reach-the-caller-ahead-of-held-frames",
        ),
        pytest.param(
            [],
            0,
            {"primary": ((START, PING, _delta("hello")),)},
            (START, PING, _delta("hello")),
            ("primary",),
            id="an-unrecoverable-stream-keeps-its-frames-in-order",
        ),
    ),
)
async def test_a_streamed_anthropic_message_recovers_the_same_way_on_both_backends(
    fallbacks: Sequence[Mapping[str, Sequence[str]]],
    num_retries: int,
    script: Script,
    expected_frames: tuple[bytes, ...],
    expected_calls: tuple[str, ...],
) -> None:
    observed: Final = [await _stream(build, _arguments(fallbacks, num_retries), script) for build in BACKENDS]

    assert observed[1][:2] == observed[0][:2]
    assert observed[1][:2] == (expected_frames, expected_calls)


async def test_a_retried_anthropic_stream_reports_its_retry_in_both_backends_headers() -> None:
    script: Final = {"primary": ((START, OVERLOADED),)}
    observed: Final = [await _stream(build, _arguments([], 2), script) for build in BACKENDS]

    assert observed[1][2] == observed[0][2]
    assert observed[1][2]["x-litellm-attempted-retries"] == 1


async def test_both_backends_raise_the_same_provider_error_when_the_fallback_fails_too() -> None:
    errors: Final[list[str]] = []  # mutable-ok: one error per backend
    for build in BACKENDS:
        with pytest.raises(Exception) as caught:  # noqa: PT011  # the type is compared across backends below
            await _stream(
                build, _arguments([{"primary": ["fb1"]}], 0), {"primary": ((START, OVERLOADED),), "fb1": (None,)}
            )
        errors.append(f"{type(caught.value).__name__}: {caught.value}")

    assert errors[1] == errors[0]
    assert "Overloaded" in errors[1]
