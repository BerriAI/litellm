"""The Rust router backend against the Python one, on the same config and the same seed."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.router_backends.rust_router import RustRouter
from litellm.router_backends.selection import select_backend
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.router import RouterRateLimitError
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse, ModelResponseStream
from tests.test_litellm_rust.support.router_backends import (
    BACKENDS,
    Backend,
    python_backend,
    router_headers,
    rust_backend,
)

pytestmark = pytest.mark.requires_rust_extension

MESSAGES: Final = ({"role": "user", "content": "hi"},)


def _deployment(
    group: str, name: str, response: object = None
) -> dict[str, object]:  # mutable-ok: PythonRouter consumes model_list dicts
    return {
        "model_name": group,
        "litellm_params": {"model": f"openai/{name}", "api_key": "k", "mock_response": response or f"from {name}"},
        "model_info": {"id": name},
    }


_python: Final = python_backend
_rust: Final = rust_backend
_headers: Final = router_headers


def _content(response: object) -> str:
    assert isinstance(response, ModelResponse)
    return str(response.choices[0].message.content)  # pyright: ignore[reportAttributeAccessIssue]  # non-streaming choice


async def _picks(backend: Backend, count: int) -> list[str]:
    return [_content(await backend.acompletion("g", MESSAGES)) for _ in range(count)]


@pytest.mark.parametrize("weights", ((None, None, None), (3, 1, 1)))
async def test_seeded_backends_pick_the_same_deployments(weights: tuple[int | None, ...]) -> None:
    model_list: Final = tuple(
        {**_deployment("g", name), "litellm_params": {**_deployment("g", name)["litellm_params"], "weight": weight}}  # pyright: ignore[reportGeneralTypeIssues]  # nested dict merge
        if weight is not None
        else _deployment("g", name)
        for name, weight in zip(("a", "b", "c"), weights)
    )

    python_picks: Final = await _picks(_python({"model_list": model_list}, 11), 12)
    rust_picks: Final = await _picks(_rust({"model_list": model_list}, 11), 12)

    assert rust_picks == python_picks
    assert len(set(python_picks)) > 1


@pytest.mark.parametrize("build", BACKENDS)
async def test_a_mock_fallback_answers_from_the_fallback_group(
    build: Callable[[Mapping[str, object], int], Backend],
) -> None:
    backend: Final = build(
        {"model_list": (_deployment("g", "a"), _deployment("h", "b")), "fallbacks": [{"g": ["h"]}]}, 1
    )

    response: Final = await backend.acompletion("g", MESSAGES, mock_testing_fallbacks=True)

    assert _content(response) == "from b"
    assert _headers(response) == {
        "x-litellm-model-group": "h",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
    }


async def test_failed_fallbacks_raise_the_same_error_from_both_backends() -> None:
    arguments: Final = {
        "model_list": (
            _deployment("g", "a", Exception("primary broke")),
            _deployment("h", "b", Exception("fallback broke")),
        ),
        "fallbacks": [{"g": ["h"]}],
        "num_retries": 1,
    }
    raised: Final[list[BaseException]] = []  # mutable-ok: collects one error per backend
    for build in BACKENDS:
        with pytest.raises(litellm.InternalServerError) as caught:
            await build(arguments, 3).acompletion("g", MESSAGES)
        raised.append(caught.value)

    python_error, rust_error = raised
    assert str(rust_error) == str(python_error)
    assert "Fallback to h also failed" in str(rust_error)
    assert (getattr(rust_error, "num_retries", None), getattr(rust_error, "max_retries", None)) == (
        getattr(python_error, "num_retries", None),
        getattr(python_error, "max_retries", None),
    )


async def test_rate_limited_deployments_cool_down_until_the_router_rejects() -> None:
    arguments: Final = {
        "model_list": (
            _deployment("g", "a", "litellm.RateLimitError"),
            _deployment("g", "b", "litellm.RateLimitError"),
        ),
        "num_retries": 0,
    }
    outcomes: Final[list[tuple[str, ...]]] = []  # mutable-ok: one sequence of outcomes per backend
    for build in BACKENDS:
        backend = build(arguments, 5)
        seen: list[str] = []  # mutable-ok: outcomes in order
        for _ in range(3):
            try:
                await backend.acompletion("g", MESSAGES)
            except RouterRateLimitError as error:
                seen.append(f"rejected:{error.type}")
            except litellm.RateLimitError:
                seen.append("rate_limited")
        outcomes.append(tuple(seen))

    assert outcomes[0] == outcomes[1]
    assert outcomes[1][-1] == "rejected:all_deployments_in_cooldown"


@pytest.mark.parametrize("build", BACKENDS)
def test_sync_completion_retries_and_answers(build: Callable[[Mapping[str, object], int], Backend]) -> None:
    backend: Final = build({"model_list": (_deployment("g", "a"),), "num_retries": 2}, 1)

    response: Final = backend.completion("g", MESSAGES)

    assert _content(response) == "from a"


def test_sync_calls_leave_the_call_counters_as_python_does() -> None:
    arguments: Final = {
        "model_list": (_deployment("g", "a"), _deployment("h", "b", Exception("broke"))),
        "num_retries": 0,
    }
    counters: Final[list[tuple[object, ...]]] = []  # mutable-ok: one reading per backend
    for build in BACKENDS:
        backend = build(arguments, 1)
        backend.completion("g", MESSAGES)
        with pytest.raises(litellm.InternalServerError):
            backend.completion("h", MESSAGES)
        counters.append(tuple(dict(getattr(backend, name)) for name in ("total_calls", "success_calls", "fail_calls")))

    assert counters[1] == counters[0]


async def test_the_facade_serves_a_supported_config_from_the_rust_backend() -> None:
    backend: Final = select_backend(
        (), {"model_list": [_deployment("g", "a")]}, rules=(RouteRule(Route.ROUTER, Rollout.RUST_REQUIRED),)
    )

    assert isinstance(backend, RustRouter)
    assert _content(await backend.acompletion("g", MESSAGES)) == "from a"
    assert backend.get_model_names() == ["g"]
    with pytest.raises(NotImplementedError, match=r"Router\.upsert_deployment"):
        _ = backend.upsert_deployment


class _FailingStream(CustomStreamWrapper):
    """A provider stream that dies before its first chunk, as a 500 mid-stream does."""

    def __init__(self, model: str) -> None:
        super().__init__(completion_stream=object(), model=model, custom_llm_provider="openai", logging_obj=MagicMock())

    def _fail(self) -> ModelResponseStream:
        raise MidStreamFallbackError(
            message=f"provider 500 from {self.model}",
            model=str(self.model),
            llm_provider="openai",
            is_pre_first_chunk=True,
            original_exception=litellm.InternalServerError(
                message=f"provider 500 from {self.model}", model=str(self.model), llm_provider="openai"
            ),
        )

    def __aiter__(self) -> _FailingStream:
        return self

    async def __anext__(self) -> ModelResponseStream:
        return self._fail()

    def __iter__(self) -> _FailingStream:
        return self

    def __next__(self) -> ModelResponseStream:
        return self._fail()


class _OkStream(_FailingStream):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._replies: Final = iter(
            (
                ModelResponseStream(choices=[{"index": 0, "delta": {"role": "assistant"}}]),
                ModelResponseStream(choices=[{"index": 0, "delta": {"content": f"ok-from-{model}"}}]),
            )
        )

    async def __anext__(self) -> ModelResponseStream:
        reply: Final = next(self._replies, None)
        if reply is None:
            raise StopAsyncIteration
        return reply

    def __next__(self) -> ModelResponseStream:
        reply: Final = next(self._replies, None)
        if reply is None:
            raise StopIteration
        return reply


def _streams(healthy: str) -> Callable[..., CustomStreamWrapper]:
    def stream(**kwargs: object) -> CustomStreamWrapper:  # kwargs-ok: litellm.completion's surface
        model: Final = str(kwargs["model"])
        return _OkStream(model) if healthy in model else _FailingStream(model)

    return stream


def _async_streams(healthy: str) -> Callable[..., object]:
    sync: Final = _streams(healthy)

    async def stream(**kwargs: object) -> CustomStreamWrapper:  # kwargs-ok: litellm.acompletion's surface
        return sync(**kwargs)

    return stream


def _stream_groups(
    fallbacks: list[dict[str, list[str]]],
) -> Mapping[str, object]:  # mutable-ok: Router's fallbacks shape
    return {
        "model_list": tuple(
            {"model_name": group, "litellm_params": {"model": f"openai/{group}-model", "api_key": "k"}}
            for group in ("primary", "fb1", "fb2")
        ),
        "fallbacks": fallbacks,
        "num_retries": 2,
    }


StreamOutcome = tuple[str, tuple[str, ...], Mapping[str, object]]


async def _astream(backend: Backend, healthy: str) -> StreamOutcome:
    with patch("litellm.acompletion", side_effect=_async_streams(healthy)) as acompletion:
        try:
            response = await backend.acompletion("primary", MESSAGES, stream=True)
            content = "".join([chunk.choices[0].delta.content or "" async for chunk in response])  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]  # stream chunks
        except litellm.InternalServerError as error:
            content = f"raised {type(error).__name__}: {error}"
            response = None
    groups: Final = tuple(str(call.kwargs["metadata"]["model_group"]) for call in acompletion.call_args_list)
    return content, groups, _headers(response)


@pytest.mark.parametrize(
    ("fallbacks", "healthy", "expected"),
    (
        pytest.param(
            [{"primary": ["fb1", "fb2"]}],
            "fb2",
            ("ok-from-openai/fb2-model", ("primary", "fb1", "fb2")),
            id="chain-of-one-group",
        ),
        pytest.param(
            [{"primary": ["fb1"]}, {"fb1": ["fb2"]}],
            "fb2",
            ("ok-from-openai/fb2-model", ("primary", "fb1", "fb2")),
            id="fallback-hop-uses-its-own-chain",
        ),
        pytest.param(
            [{"primary": ["fb1", "fb2"]}],
            "nowhere",
            (
                "raised InternalServerError: litellm.InternalServerError: provider 500 from openai/fb2-model",
                ("primary", "fb1", "fb2"),
            ),
            id="every-group-fails",
        ),
    ),
)
async def test_a_stream_failing_before_content_falls_back_the_same_way_on_both_backends(
    fallbacks: list[dict[str, list[str]]],  # mutable-ok: Router's fallbacks shape
    healthy: str,
    expected: tuple[str, tuple[str, ...]],
) -> None:
    python: Final = await _astream(_python(_stream_groups(fallbacks), 1), healthy)
    rust: Final = await _astream(_rust(_stream_groups(fallbacks), 1), healthy)

    assert rust == python
    assert rust[:2] == expected


def test_a_sync_stream_failing_before_content_falls_back_like_the_async_path() -> None:
    """Python's sync wrapper reruns the same group instead of falling back, and recurses until
    `RecursionError` while the group keeps failing; the Rust backend applies the async rule."""
    with patch("litellm.completion", side_effect=_streams("fb2")) as completion:
        response = _rust(_stream_groups([{"primary": ["fb1", "fb2"]}]), 1).completion("primary", MESSAGES, stream=True)
        content = "".join(chunk.choices[0].delta.content or "" for chunk in response)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType,reportGeneralTypeIssues]  # stream chunks

    assert content == "ok-from-openai/fb2-model"
    assert [call.kwargs["metadata"]["model_group"] for call in completion.call_args_list] == ["primary", "fb1", "fb2"]
