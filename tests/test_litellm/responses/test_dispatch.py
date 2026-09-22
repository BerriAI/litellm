import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest

import litellm
from litellm.responses import dispatch as responses_dispatch
from litellm.responses import main as python_responses
from litellm.responses.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.responses.entrypoints import (
    NATIVE_ARESPONSES,
    NATIVE_RESPONSES,
    LiteLLMResponsesRequest,
    NativeAresponses,
    NativeResponses,
)
from litellm.types.llms.openai import ResponsesAPIResponse

INPUT: Final = [{"role": "user", "content": "hi"}]
PYTHON_RULES: Final = ()
RUST_RULES: Final = (RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),)


def _response(model: str = "gpt-4o") -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_test", object="response", created_at=0, model=model, output=[], status="completed"
    )


def responses_binding(native: NativeResponses | None) -> NativeBinding[NativeResponses]:
    binding: Final[NativeBinding[NativeResponses]] = NativeBinding("responses", validate=lambda _: None)
    binding.override(native)
    return binding


def aresponses_binding(native: NativeAresponses | None) -> NativeBinding[NativeAresponses]:
    binding: Final[NativeBinding[NativeAresponses]] = NativeBinding("aresponses", validate=lambda _: None)
    binding.override(native)
    return binding


def test_public_signature_is_the_legacy_signature() -> None:
    public_responses: Final = cast(Callable[..., object], litellm.responses)
    legacy_responses: Final = cast(Callable[..., object], python_responses.responses)
    public_aresponses: Final = cast(Callable[..., object], litellm.aresponses)
    legacy_aresponses: Final = cast(Callable[..., object], python_responses.aresponses)
    assert inspect.signature(public_responses) == inspect.signature(legacy_responses)
    assert inspect.signature(public_aresponses) == inspect.signature(legacy_aresponses)


def test_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (INPUT, "gpt-4o")
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "litellm_metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = _response()

    def python(*call_args: object, **call_kwargs: object) -> ResponsesAPIResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return response

    def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        pytest.fail("Python-only dispatch must not call native")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=responses_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
            rules=PYTHON_RULES,
        )
        is response
    )
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[0] is INPUT
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "litellm_metadata": metadata}


@pytest.mark.asyncio
async def test_async_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (INPUT, "gpt-4o")
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "litellm_metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = _response()

    async def python(
        *call_args: object,
        **call_kwargs: object,  # kwargs-ok: records call shape
    ) -> ResponsesAPIResponse:
        captured.append((call_args, call_kwargs))
        return response

    async def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=aresponses_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=PYTHON_RULES,
    )
    assert result is response
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[0] is INPUT
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "litellm_metadata": metadata}


def test_native_receives_normalized_request_and_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    extra_headers: Final = {"x-test": "1"}
    args: Final[tuple[object, ...]] = (INPUT, "anthropic/claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "base_url": "https://example.invalid",
        "extra_headers": extra_headers,
        "custom_llm_provider": "anthropic",
        "litellm_metadata": metadata,
    }
    captured: Final[list[tuple[LiteLLMResponsesRequest, tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = _response("anthropic/claude-sonnet-4-5")

    def python(*call_args: object, **call_kwargs: object) -> ResponsesAPIResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        captured.append((request, args, kwargs))
        return response

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=responses_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert result is response
    assert request.model == "anthropic/claude-sonnet-4-5"
    assert request.input is INPUT
    assert request.stream is True
    assert request.api_key == "sk-test"
    assert request.api_base == "https://example.invalid"
    assert request.custom_llm_provider == "anthropic"
    assert request.extra_headers is extra_headers
    assert request.kwargs == {
        "api_key": "sk-test",
        "base_url": "https://example.invalid",
        "litellm_metadata": metadata,
    }
    assert request.kwargs["litellm_metadata"] is metadata
    assert call_args == args
    assert call_args[0] is INPUT
    assert call_kwargs == kwargs
    assert call_kwargs["extra_headers"] is extra_headers
    assert call_kwargs["litellm_metadata"] is metadata


def test_internal_async_marker_bypasses_native() -> None:
    args: Final[tuple[object, ...]] = (INPUT, "gpt-4o")
    kwargs: Final[Mapping[str, object]] = {"aresponses": True}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = _response()

    def python(*call_args: object, **call_kwargs: object) -> ResponsesAPIResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return response

    def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        pytest.fail("aresponses' inner responses call must stay on Python")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=responses_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
            rules=RUST_RULES,
        )
        is response
    )
    assert captured == [(args, kwargs)]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        ((INPUT, "gpt-4o"), {"model": "duplicate"}),
        ((), {}),
    ),
)
def test_binding_errors_delegate_unchanged_to_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = _response()

    def python(*call_args: object, **call_kwargs: object) -> ResponsesAPIResponse:  # kwargs-ok: records invalid call
        captured.append((call_args, call_kwargs))
        return response

    def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        pytest.fail("Binding failures must be delegated to Python")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=responses_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request, call_args, call_kwargs),
            rules=RUST_RULES,
        )
        is response
    )
    assert captured == [(args, kwargs)]


def test_public_responses_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[LiteLLMResponsesRequest]] = []
    expected: Final = _response()

    def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        captured.append(request)
        return expected

    NATIVE_RESPONSES.override(native)
    monkeypatch.setattr(catalog, "RULES", RUST_RULES)
    public_responses: Final = cast(Callable[..., ResponsesAPIResponse], litellm.responses)
    try:
        result: Final = public_responses(input=INPUT, model="gpt-4o")
    finally:
        NATIVE_RESPONSES.reset()
    assert result is expected
    assert [request.model for request in captured] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_public_aresponses_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[LiteLLMResponsesRequest]] = []
    expected: Final = _response()

    async def native(
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse:
        captured.append(request)
        return expected

    NATIVE_ARESPONSES.override(native)
    monkeypatch.setattr(catalog, "RULES", RUST_RULES)
    public_aresponses: Final = cast(Callable[..., Awaitable[ResponsesAPIResponse]], litellm.aresponses)
    try:
        result: Final = await public_aresponses(input=INPUT, model="gpt-4o")
    finally:
        NATIVE_ARESPONSES.reset()
    assert result is expected
    assert [request.model for request in captured] == ["gpt-4o"]


def test_responses_with_retries_uses_the_dispatch_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Final[list[Mapping[str, object]]] = []
    expected: Final = _response()

    def dispatch_responses(*args: object, **kwargs: object) -> ResponsesAPIResponse:  # kwargs-ok: records call shape
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(responses_dispatch, "responses", dispatch_responses)
    retry: Final = cast(Callable[..., ResponsesAPIResponse], litellm.responses_with_retries)
    result: Final = retry(input=INPUT, model="gpt-4o", num_retries=1)
    assert result is expected
    assert calls[0]["num_retries"] == 0
    assert calls[0]["max_retries"] == 0
