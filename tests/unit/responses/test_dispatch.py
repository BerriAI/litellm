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
from litellm.rust_bridge.catalog import Decision, Route, RouteContext, Rust
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.responses.entrypoints import NATIVE_ARESPONSES, NATIVE_RESPONSES
from litellm.types.llms.openai import ResponsesAPIResponse

INPUT: Final = [{"role": "user", "content": "hi"}]


def required_everywhere(_context: RouteContext) -> Decision:
    return Rust(required=True)


def _response(model: str = "gpt-4o") -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_test", object="response", created_at=0, model=model, output=[], status="completed"
    )


def test_public_signature_is_the_legacy_signature() -> None:
    public_responses: Final = cast(Callable[..., object], litellm.responses)
    legacy_responses: Final = cast(Callable[..., object], python_responses.responses)
    public_aresponses: Final = cast(Callable[..., object], litellm.aresponses)
    legacy_aresponses: Final = cast(Callable[..., object], python_responses.aresponses)
    assert inspect.signature(public_responses) == inspect.signature(legacy_responses)
    assert inspect.signature(public_aresponses) == inspect.signature(legacy_aresponses)


@pytest.mark.parametrize("dispatch", (_DISPATCH, _ADISPATCH), ids=("responses", "aresponses"))
def test_request_binds_positional_and_keyword_arguments_onto_the_legacy_signature(dispatch: PublicDispatch) -> None:
    include: Final = ["reasoning.encrypted_content"]
    extra_headers: Final = {"x-test": "1"}
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "custom_llm_provider": "anthropic",
        "extra_headers": extra_headers,
    }

    request: Final = dispatch.request((INPUT, "anthropic/claude-sonnet-4-5", include, "Be brief", 16), kwargs)

    assert request is not None
    assert request.bound["input"] is INPUT
    assert request.bound["model"] == "anthropic/claude-sonnet-4-5"
    assert request.bound["include"] is include
    assert request.bound["instructions"] == "Be brief"
    assert request.bound["max_output_tokens"] == 16
    assert request.bound["stream"] is True
    assert request.bound["api_key"] == "sk-test"
    assert request.bound["extra_headers"] is extra_headers
    assert request.kwargs is kwargs
    assert dispatch.context(request) == RouteContext(
        Route.RESPONSES, provider="anthropic", model="anthropic/claude-sonnet-4-5"
    )


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        pytest.param((INPUT, None), {}, id="unnamed-model"),
        pytest.param((INPUT, "gpt-4o"), {"model": "duplicate"}, id="does-not-bind"),
        pytest.param((INPUT, "gpt-4o"), {"aresponses": True}, id="internal-async-hop"),
    ),
)
def test_request_stays_on_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    assert _DISPATCH.request(args, kwargs) is None


def test_public_responses_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = _response()

    def native(request: NativeCall) -> ResponsesAPIResponse:
        captured.append(request)
        return expected

    NATIVE_RESPONSES.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_responses: Final = cast(Callable[..., ResponsesAPIResponse], litellm.responses)
    try:
        result: Final = public_responses(input=INPUT, model="gpt-4o")
    finally:
        NATIVE_RESPONSES.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["gpt-4o"]


@pytest.mark.asyncio
async def test_public_aresponses_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = _response()

    async def native(request: NativeCall) -> ResponsesAPIResponse:
        captured.append(request)
        return expected

    NATIVE_ARESPONSES.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_aresponses: Final = cast(Callable[..., Awaitable[ResponsesAPIResponse]], litellm.aresponses)
    try:
        result: Final = await public_aresponses(input=INPUT, model="gpt-4o")
    finally:
        NATIVE_ARESPONSES.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["gpt-4o"]


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
