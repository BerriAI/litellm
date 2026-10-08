import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.anthropic.pass_through.messages import handler as python_messages
from litellm.messages import dispatch
from litellm.messages.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Decision, Python, RouteContext, Rust
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES, NativeAmessages, NativeMessages
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]
PYTHON: Final = Python("test keeps the call on Python")
REQUIRED: Final = Rust(required=True)


def required_everywhere(_context: RouteContext) -> Decision:
    return REQUIRED


def messages_binding(native: NativeMessages | None) -> NativeBinding[NativeMessages]:
    binding: Final[NativeBinding[NativeMessages]] = NativeBinding("anthropic_messages_handler", validate=lambda _: None)
    binding.override(native)
    return binding


def amessages_binding(native: NativeAmessages | None) -> NativeBinding[NativeAmessages]:
    binding: Final[NativeBinding[NativeAmessages]] = NativeBinding("anthropic_messages", validate=lambda _: None)
    binding.override(native)
    return binding


def response(model: str = "claude-sonnet-4-5") -> AnthropicMessagesResponse:
    return AnthropicMessagesResponse(id="msg_test", type="message", role="assistant", model=model, content=[])


def test_public_signature_is_the_legacy_signature() -> None:
    public_messages: Final = cast(Callable[..., object], litellm.anthropic_messages_handler)
    legacy_messages: Final = cast(Callable[..., object], python_messages.anthropic_messages_handler)
    public_amessages: Final = cast(Callable[..., object], litellm.anthropic_messages)
    legacy_amessages: Final = cast(Callable[..., object], python_messages.anthropic_messages)
    assert inspect.signature(public_messages) == inspect.signature(legacy_messages)
    assert inspect.signature(public_amessages) == inspect.signature(legacy_amessages)


def test_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (16, MESSAGES, "claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "litellm_metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        policy=PYTHON,
    )
    assert result is expected
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "litellm_metadata": metadata}


@pytest.mark.asyncio
async def test_async_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (16, MESSAGES, "claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "litellm_metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    async def python(
        *call_args: object,
        **call_kwargs: object,  # kwargs-ok: records call shape
    ) -> AnthropicMessagesResponse:
        captured.append((call_args, call_kwargs))
        return expected

    async def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        policy=PYTHON,
    )
    assert result is expected
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "litellm_metadata": metadata}


def test_native_receives_normalized_request_and_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = (16, MESSAGES, "anthropic/claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "api_base": "https://example.invalid",
        "custom_llm_provider": "anthropic",
        "litellm_metadata": metadata,
    }
    captured: Final[list[tuple[NativeCall, tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response("anthropic/claude-sonnet-4-5")

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        args: Final = request.args
        kwargs: Final = request.kwargs
        captured.append((request, args, kwargs))
        return expected

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        policy=REQUIRED,
    )
    assert result is expected
    request, call_args, call_kwargs = captured[0]
    assert request.bound["model"] == "anthropic/claude-sonnet-4-5"
    assert request.bound["messages"] is MESSAGES
    assert request.bound["max_tokens"] == 16
    assert request.bound["stream"] is True
    assert request.bound["api_key"] == "sk-test"
    assert request.bound["api_base"] == "https://example.invalid"
    assert request.bound["custom_llm_provider"] == "anthropic"
    assert request.kwargs == kwargs
    assert request.kwargs["litellm_metadata"] is metadata
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata


def test_internal_async_marker_bypasses_native() -> None:
    args: Final[tuple[object, ...]] = (16, MESSAGES, "claude-sonnet-4-5")
    kwargs: Final[Mapping[str, object]] = {"is_async": True}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        pytest.fail("The async handler's inner sync call must stay on Python")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        policy=REQUIRED,
    )
    assert result is expected
    assert captured == [(args, kwargs)]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        ((16, MESSAGES, "claude-sonnet-4-5"), {"model": "duplicate"}),
        ((), {}),
    ),
)
def test_binding_errors_delegate_to_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    expected: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:
        captured.append((call_args, call_kwargs))
        return expected

    def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        pytest.fail("Binding failures must be delegated to Python")

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        policy=REQUIRED,
    )
    assert result is expected
    assert captured == [(args, kwargs)]


def test_anthropic_create_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        captured.append(request)
        return expected

    NATIVE_MESSAGES.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_create: Final = cast(Callable[..., AnthropicMessagesResponse], litellm.anthropic.create)
    try:
        result: Final = public_create(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5")
    finally:
        NATIVE_MESSAGES.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_anthropic_acreate_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    async def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        captured.append(request)
        return expected

    NATIVE_AMESSAGES.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_acreate: Final = cast(Callable[..., Awaitable[AnthropicMessagesResponse]], litellm.anthropic.acreate)
    try:
        result: Final = await public_acreate(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5")
    finally:
        NATIVE_AMESSAGES.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_public_anthropic_messages_keeps_the_python_result() -> None:
    response: Final = await litellm.anthropic_messages(
        model="anthropic/claude-sonnet-4-5", messages=MESSAGES, max_tokens=10, mock_response="ok"
    )

    assert isinstance(response, dict)
    content: Final = TypeAdapter(list[dict[str, object]]).validate_python(response.get("content", []))
    assert content[0]["text"] == "ok"


def test_sync_messages_request_projects_public_arguments() -> None:
    policy: Final = Rust(required=True)
    expected: Final = AnthropicMessagesResponse(model="claude-test")

    def native(request: NativeCall) -> AnthropicMessagesResponse:
        assert request.bound["model"] == "claude-test"
        assert request.bound["messages"] == MESSAGES
        assert request.bound["max_tokens"] == 10
        assert request.bound["custom_llm_provider"] == "anthropic"
        return expected

    binding: Final[NativeBinding[NativeMessages]] = NativeBinding("messages", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {
            "model": "claude-test",
            "messages": MESSAGES,
            "max_tokens": 10,
            "custom_llm_provider": "anthropic",
        },
        python=lambda *args, **kwargs: pytest.fail("required native route must handle this call"),
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request),
        policy=policy,
    )

    assert response is expected


def test_messages_binding_error_delegates_unchanged_to_python() -> None:
    policy: Final = Rust(required=True)
    expected: Final = AnthropicMessagesResponse(model="claude-test")

    def python(*args: object, **kwargs: object) -> AnthropicMessagesResponse:
        return expected

    def native(request: NativeCall) -> AnthropicMessagesResponse:
        pytest.fail("a call without max_tokens cannot project a request and must stay on Python")

    binding: Final[NativeBinding[NativeMessages]] = NativeBinding("messages", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {"model": "claude-test", "messages": MESSAGES, "custom_llm_provider": "anthropic"},
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request),
        policy=policy,
    )

    assert response is expected


@pytest.mark.asyncio
async def test_async_messages_falls_back_after_native_declines() -> None:
    from litellm.rust_bridge.bindings import native_exception_types

    native_types: Final = native_exception_types()
    if native_types is None:
        pytest.skip("native bridge is unavailable")
    declined, _ = native_types
    expected: Final = AnthropicMessagesResponse(model="claude-test")
    policy: Final = Rust()

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        raise declined("unsupported")

    async def python(*args: object, **kwargs: object) -> AnthropicMessagesResponse:
        return expected

    binding: Final[NativeBinding[NativeAmessages]] = NativeBinding("amessages", validate=lambda _: None)
    binding.override(native)
    response: Final = await dispatch._ADISPATCH.arun(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {"model": "claude-test", "messages": MESSAGES, "max_tokens": 10},
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request),
        policy=policy,
    )

    assert response is expected


def test_internal_is_async_marker_bypasses_native() -> None:
    policy: Final = Rust(required=True)
    expected: Final = AnthropicMessagesResponse(model="claude-test")

    def python(*args: object, **kwargs: object) -> AnthropicMessagesResponse:
        return expected

    def native(request: NativeCall) -> AnthropicMessagesResponse:
        pytest.fail("anthropic_messages' inner handler call must stay on Python")

    binding: Final[NativeBinding[NativeMessages]] = NativeBinding("messages", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        (),
        {
            "model": "claude-test",
            "messages": MESSAGES,
            "max_tokens": 10,
            "custom_llm_provider": "anthropic",
            "is_async": True,
        },
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request),
        policy=policy,
    )

    assert response is expected
