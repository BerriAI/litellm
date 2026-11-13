import inspect
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import httpx
import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.constants import COMPLETION_HTTP_FALLBACK_SECONDS
from litellm.exceptions import UnsupportedParamsError
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.llms.anthropic.messages.native import handler as python_messages
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.anthropic.messages.native import interceptors as messages_interceptors
from litellm.llms.anthropic.messages.native.interceptors import MessagesInterceptor
from litellm.messages import dispatch
from litellm.messages.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.proxy import proxy_server
from litellm.rust_bridge import catalog
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import MESSAGES_PYTHON_FEATURES, RULES, Feature, Route, RouteRule, Rules, Target
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES, NativeAmessages, NativeMessages
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.routing import routed

MESSAGES: Final = [{"role": "user", "content": "hi"}]
PYTHON_RULES: Final[Rules] = ()
RUST_RULES: Final[Rules] = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)


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
        rules=PYTHON_RULES,
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
        rules=PYTHON_RULES,
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
        rules=RUST_RULES,
    )
    assert result is expected
    request, call_args, call_kwargs = captured[0]
    assert request.resolved["model"] == "anthropic/claude-sonnet-4-5"
    assert request.resolved["messages"] is MESSAGES
    assert request.resolved["max_tokens"] == 16
    assert request.resolved["stream"] is True
    assert request.resolved["api_key"] == "sk-test"
    assert request.resolved["api_base"] == "https://example.invalid"
    assert request.resolved["custom_llm_provider"] == "anthropic"
    assert request.kwargs == kwargs
    assert request.kwargs["litellm_metadata"] is metadata
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["litellm_metadata"] is metadata


@pytest.mark.parametrize(
    ("kwargs", "expected_timeout"),
    (
        pytest.param({"stream": True, "timeout": 30.0, "stream_timeout": 0.5}, 0.5, id="stream-timeout-bounds-stream"),
        pytest.param(
            {"stream": False, "timeout": 30.0, "stream_timeout": 0.5}, 30.0, id="stream-timeout-ignored-unary"
        ),
        pytest.param({"request_timeout": 5}, 5.0, id="request-timeout"),
        pytest.param({"timeout": 30.0, "request_timeout": 5}, 30.0, id="timeout-before-request-timeout"),
        pytest.param(
            {"stream": True, "request_timeout": 5, "stream_timeout": 0.5}, 0.5, id="stream-timeout-over-request"
        ),
        pytest.param({"timeout": httpx.Timeout(60.0, connect=2.0)}, 60.0, id="httpx-timeout-read"),
        pytest.param(
            {"timeout": httpx.Timeout(None, connect=2.0)}, COMPLETION_HTTP_FALLBACK_SECONDS, id="httpx-timeout-no-read"
        ),
    ),
)
def test_native_timeout_resolves_like_python(kwargs: Mapping[str, object], expected_timeout: float) -> None:
    captured: Final[list[NativeCall]] = []

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(request: NativeCall) -> AnthropicMessagesResponse:
        captured.append(request)
        return response()

    _ = _DISPATCH.run(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        kwargs,
        python=python,
        binding=messages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=RUST_RULES,
    )
    assert captured[0].resolved["timeout"] == expected_timeout


@pytest.mark.parametrize(
    ("model", "kwargs", "passes_through"),
    (
        ("vertex_ai/claude-sonnet-4-5@20250929", {}, True),
        ("claude-sonnet-4-5@20250929", {"custom_llm_provider": "vertex_ai"}, True),
        ("deepseek/deepseek-v4-flash", {}, True),
        ("openai/gpt-5", {"model_info": {"supported_endpoints": ["/v1/messages"]}}, True),
        ("vertex_ai/gemini-2.5-pro", {}, False),
        ("gemini-2.5-pro", {"custom_llm_provider": "vertex_ai"}, False),
        ("openai/gpt-5", {}, False),
        ("groq/llama-3.3-70b-versatile", {}, False),
        ("unknown-provider/unknown-model", {}, False),
    ),
)
def test_only_requests_that_pass_through_to_a_messages_api_reach_a_messages_target_rule(
    model: str, kwargs: Mapping[str, object], passes_through: bool
) -> None:
    python_result: Final = response(model)
    native_result: Final = response(model)

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: fallback
        return python_result

    result: Final = _DISPATCH.run(
        (16, MESSAGES, model),
        kwargs,
        python=python,
        binding=messages_binding(lambda request: native_result),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED, targets=frozenset({Target.MESSAGES})),),
    )
    assert result is (native_result if passes_through else python_result)


DANGEROUS_TOOL_USE: Final = [{"type": "dangerous_tool_use", "classifier_context": {"permission_mode": "auto"}}]
CLASSIFIER_SETTINGS: Final = {"safeguards_classifier_model": "gpt-oss-120b"}


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
        rules=RUST_RULES,
    )
    assert result is expected
    assert captured == [(args, kwargs)]


def test_anthropic_create_routes_through_dispatch() -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        captured.append(request)
        return expected

    NATIVE_MESSAGES.override(native)
    public_create: Final = cast(Callable[..., AnthropicMessagesResponse], litellm.anthropic.create)
    try:
        result: Final = public_create(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5")
    finally:
        NATIVE_MESSAGES.reset()
    assert result is expected
    assert [request.resolved["model"] for request in captured] == ["claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_anthropic_acreate_routes_through_dispatch() -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    async def native(
        request: NativeCall,
    ) -> AnthropicMessagesResponse:
        captured.append(request)
        return expected

    NATIVE_AMESSAGES.override(native)
    public_acreate: Final = cast(Callable[..., Awaitable[AnthropicMessagesResponse]], litellm.anthropic.acreate)
    try:
        result: Final = await public_acreate(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5")
    finally:
        NATIVE_AMESSAGES.reset()
    assert result is expected
    assert [request.resolved["model"] for request in captured] == ["claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_truthy_is_async_hands_off_to_the_async_entry_like_true() -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = response()

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        captured.append(request)
        return expected

    def sync_native(request: NativeCall) -> AnthropicMessagesResponse:
        pytest.fail("a truthy is_async must hand off to the async entry, as the Python handler does")

    NATIVE_AMESSAGES.override(native)
    NATIVE_MESSAGES.override(sync_native)
    public_create: Final = cast(Callable[..., object], litellm.anthropic.create)
    try:
        pending: Final = public_create(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5", is_async=1)
        assert inspect.isawaitable(pending)
        assert await pending is expected
        reference: Final = public_create(max_tokens=16, messages=MESSAGES, model="claude-sonnet-4-5", is_async=True)
        assert inspect.isawaitable(reference)
        assert await reference is expected
    finally:
        NATIVE_AMESSAGES.reset()
        NATIVE_MESSAGES.reset()
    assert [dict(request.resolved) for request in captured] == [dict(captured[1].resolved)] * 2


def test_sync_messages_request_projects_public_arguments() -> None:
    rules: Final[Rules] = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)
    expected: Final = AnthropicMessagesResponse(model="claude-test")

    def native(request: NativeCall) -> AnthropicMessagesResponse:
        assert request.resolved["model"] == "claude-test"
        assert request.resolved["messages"] == MESSAGES
        assert request.resolved["max_tokens"] == 10
        assert request.resolved["custom_llm_provider"] == "anthropic"
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
        rules=rules,
    )

    assert response is expected


def test_messages_binding_error_delegates_unchanged_to_python() -> None:
    rules: Final[Rules] = (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),)
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
        rules=rules,
    )

    assert response is expected


class _Rewrite(CustomLogger):
    async def async_pre_request_hook(self, model: str, messages: list, kwargs: dict) -> dict | None:  # pyright: ignore[reportMissingTypeArgument]  # mirrors the CustomLogger signature
        return kwargs


class _Agentic(CustomLogger):
    async def async_should_run_agentic_loop(self, *args: object, **kwargs: object) -> tuple[bool, dict]:  # pyright: ignore[reportIncompatibleMethodOverride, reportMissingTypeArgument]  # test double only needs the override
        return False, {}


class _ShortCircuit(WebSearchInterceptionLogger):
    async def try_short_circuit_search(self, *args: object, **kwargs: object) -> dict[str, object] | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double only needs the override
        return None


ADVISOR_TOOL: Final = {"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "name": "advisor", "model": "claude-opus-5-5"}
STRICT_TOOL: Final = {"name": "a", "input_schema": {"type": "object"}, "strict": True}


@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param({"shared_session": object()}, id="proxy-shared-session"),
        pytest.param({"stream": True, "timeout": 600.0, "stream_timeout": 600.0}, id="proxy-stream-timeout"),
        pytest.param({"request_timeout": 5}, id="request-timeout"),
        pytest.param({"timeout": httpx.Timeout(60.0, connect=2.0)}, id="httpx-timeout"),
    ),
)
def test_proxy_request_shape_stays_native(kwargs: Mapping[str, object]) -> None:
    rules: Final[Rules] = (
        RouteRule(
            Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({Feature.CALLER_CLIENT, Feature.TIMEOUT_VARIANTS})
        ),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    native_result: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("A proxy-shaped request must not fall back to Python")

    served: Final = _DISPATCH.run(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        kwargs,
        python=python,
        binding=messages_binding(lambda request: native_result),
        native=lambda hook, request, call_args, native_kwargs: hook(request),
        rules=rules,
    )
    assert served is native_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block",
    (
        pytest.param({"type": "thinking", "thinking": "x", "signature": "s"}, id="thinking"),
        pytest.param({"type": "redacted_thinking", "data": "x"}, id="redacted-thinking"),
    ),
)
async def test_replayed_thinking_is_served_natively_under_the_shipped_rules(
    monkeypatch: pytest.MonkeyPatch, block: Mapping[str, object]
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "cache", None)
    replay: Final = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [block, {"type": "text", "text": "hello"}]},
        {"role": "user", "content": "again"},
    ]
    native_result: Final = response()
    python_result: Final = response()

    async def python(*call_args: object, **python_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        return python_result

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        return native_result

    served: Final = await _ADISPATCH.arun(
        (16, replay, "anthropic/claude-sonnet-4-5"),
        {"api_key": "sk-ant-test"},
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, call_args, native_kwargs: hook(request),
        rules=RULES,
    )

    assert served is native_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "global_drop_params"),
    (
        pytest.param({"drop_params": True, "tools": [STRICT_TOOL]}, False, id="drop-params-kwarg"),
        pytest.param({"tools": [STRICT_TOOL]}, True, id="drop-params-global"),
        pytest.param({"tools": [STRICT_TOOL]}, False, id="drop-params-unset"),
    ),
)
async def test_strict_tools_are_served_natively_under_the_shipped_rules(
    monkeypatch: pytest.MonkeyPatch, kwargs: Mapping[str, object], global_drop_params: bool
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    native_result: Final = response()

    async def python(*call_args: object, **python_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        pytest.fail("strict tools stay on the native path")

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        return native_result

    served: Final = await _ADISPATCH.arun(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        kwargs,
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, call_args, native_kwargs: hook(request),
        rules=RULES,
    )

    assert served is native_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param({}, id="reads-and-writes"),
        pytest.param({"cache": {"no-store": True}}, id="no-store"),
        pytest.param({"cache": {"ttl": 60, "namespace": "team"}}, id="ttl-and-namespace"),
        pytest.param({"stream": True}, id="stream"),
    ),
)
async def test_a_configured_response_cache_stays_native(
    monkeypatch: pytest.MonkeyPatch, kwargs: Mapping[str, object]
) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=MESSAGES_PYTHON_FEATURES),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    native_result: Final = response()
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "cache", Cache())

    async def python(*call_args: object, **python_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        pytest.fail("Rust reads and writes the configured response cache")

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        return native_result

    served: Final = await _ADISPATCH.arun(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        kwargs,
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, call_args, native_kwargs: hook(request),
        rules=rules,
    )

    assert served is native_result


@pytest.mark.asyncio
async def test_modify_params_stays_native(monkeypatch: pytest.MonkeyPatch) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=catalog.MESSAGES_PYTHON_FEATURES),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    native_result: Final = response()
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "modify_params", True)

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("modify_params must not fall back to Python")

    async def apython(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        pytest.fail("modify_params must not fall back to Python")

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        return native_result

    call_args: Final = (200_000, MESSAGES, "anthropic/claude-sonnet-4-5")
    served: Final = _DISPATCH.run(
        call_args,
        {},
        python=python,
        binding=messages_binding(lambda request: native_result),
        native=lambda hook, request, args, kwargs: hook(request),
        rules=rules,
    )
    aserved: Final = await _ADISPATCH.arun(
        call_args,
        {},
        python=apython,
        binding=amessages_binding(native),
        native=lambda hook, request, args, kwargs: hook(request),
        rules=rules,
    )

    assert (served, aserved) == (native_result, native_result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param({"mock_response": "hi"}, id="unary"),
        pytest.param({"mock_response": "hi", "stream": True}, id="streamed"),
    ),
)
async def test_mock_response_is_served_natively(monkeypatch: pytest.MonkeyPatch, kwargs: Mapping[str, object]) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({Feature.MOCK_RESPONSE})),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    python_result: Final = response()
    native_result: Final = response()

    async def python(*call_args: object, **python_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        return python_result

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        return native_result

    monkeypatch.setattr(litellm, "callbacks", [])
    served_async: Final = await _ADISPATCH.arun(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        kwargs,
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, args, native_kwargs: hook(request),
        rules=rules,
    )
    served_sync: Final = _DISPATCH.run(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        kwargs,
        python=lambda *call_args, **python_kwargs: python_result,
        binding=messages_binding(lambda request: native_result),
        native=lambda hook, request, args, native_kwargs: hook(request),
        rules=rules,
    )
    assert served_async is native_result
    assert served_sync is native_result


class _Claims(MessagesInterceptor):
    def can_handle(self, tools: Sequence[Mapping[str, object]] | None, custom_llm_provider: str | None) -> bool:
        return custom_llm_provider == "anthropic"

    async def handle(self, **kwargs: object) -> AnthropicMessagesResponse:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: test double only needs the override
        return response()


@pytest.mark.asyncio
async def test_a_call_a_registered_interceptor_owns_runs_on_rust(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str]
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(messages_interceptors, "_interceptors", [_Claims()])

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Rust hands the call to the interceptor itself")

    with rebound(dispatch, "_PYTHON_AMESSAGES", python):
        result: Final = await _public(True, "anthropic/claude-sonnet-4-5", {"tools": [ADVISOR_TOOL]})

    assert isinstance(result, dict)
    assert shipped_native == ["async"]


def test_a_federated_request_stays_native(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({Feature.WORKLOAD_IDENTITY})),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    native_result: Final = response()

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Rust mints the federation token itself")

    served: Final = _DISPATCH.run(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        {"anthropic_federation_rule_id": "rule-1", "anthropic_organization_id": "org-1"},
        python=python,
        binding=messages_binding(lambda request: native_result),
        native=lambda hook, request, call_args, native_kwargs: hook(request),
        rules=rules,
    )
    assert served is native_result


def _shipped_native(calls: list[str]) -> None:  # mutable-ok: records which native entry ran
    def native(request: NativeCall) -> AnthropicMessagesResponse:
        calls.append("sync")
        return response()

    async def anative(request: NativeCall) -> AnthropicMessagesResponse:
        calls.append("async")
        return response()

    NATIVE_MESSAGES.override(native)
    NATIVE_AMESSAGES.override(anative)


@pytest.fixture
def shipped_native() -> Iterator[list[str]]:
    calls: Final[list[str]] = []
    _shipped_native(calls)
    with routed(*RULES):
        yield calls
    NATIVE_MESSAGES.reset()
    NATIVE_AMESSAGES.reset()


async def _public(asynchronous: bool, model: str, kwargs: Mapping[str, object]) -> object:
    call: Final = {"max_tokens": 16, "messages": MESSAGES, "model": model, "api_key": "sk-test", **kwargs}
    if asynchronous:
        return await dispatch.anthropic_messages(**call)
    return dispatch.anthropic_messages_handler(**call)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
async def test_anthropic_runs_on_rust_without_the_opt_in(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    monkeypatch.setattr(litellm, "callbacks", [])

    result: Final = await _public(asynchronous, "anthropic/claude-sonnet-4-5", {})

    assert isinstance(result, dict)
    assert shipped_native == ["async" if asynchronous else "sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize(
    "model", ("anthropic/parity-tool-model", "anthropic/claude-sonnet-4-5"), ids=("non-claude", "claude")
)
async def test_proxy_classified_safeguards_run_on_rust(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool, model: str
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(proxy_server, "general_settings", CLASSIFIER_SETTINGS)

    result: Final = await _public(asynchronous, model, {"safeguards": DANGEROUS_TOOL_USE})

    assert isinstance(result, dict)
    assert shipped_native == ["async" if asynchronous else "sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize(
    "thread",
    (
        pytest.param({"type": "create"}, id="create"),
        pytest.param({"type": "continue", "previous_message_id": "msg_1"}, id="continue"),
    ),
)
async def test_a_message_thread_runs_on_rust(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool, thread: Mapping[str, str]
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])

    result: Final = await _public(asynchronous, "anthropic/claude-sonnet-4-5", {"thread": dict(thread)})

    assert isinstance(result, dict)
    assert shipped_native == ["async" if asynchronous else "sync"]


@pytest.mark.asyncio
async def test_anthropic_runs_pre_request_hooks_on_rust(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str]
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_Rewrite()])

    result: Final = await _public(True, "anthropic/claude-sonnet-4-5", {})

    assert isinstance(result, dict)
    assert shipped_native == ["async"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
async def test_anthropic_runs_a_web_search_short_circuit_on_rust(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_ShortCircuit()])

    result: Final = await _public(asynchronous, "anthropic/claude-sonnet-4-5", {})

    assert isinstance(result, dict)
    assert shipped_native == ["async" if asynchronous else "sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize(
    ("kwargs", "settings"),
    (
        pytest.param({"client": object()}, {}, id="caller-client"),
        pytest.param({"client": AsyncHTTPHandler()}, {}, id="caller-async-handler"),
        pytest.param({}, {"ssl_security_level": "DEFAULT@SECLEVEL=1"}, id="tls-security-level"),
        pytest.param({}, {"ssl_ecdh_curve": "secp521r1"}, id="tls-ecdh-curve"),
    ),
)
async def test_anthropic_serves_the_http_transport_features_on_rust(
    monkeypatch: pytest.MonkeyPatch,
    shipped_native: list[str],
    asynchronous: bool,
    kwargs: Mapping[str, object],
    settings: Mapping[str, object],
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    for name in ("SSL_SECURITY_LEVEL", "SSL_ECDH_CURVE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setattr(litellm, name, value)

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("anthropic has no Python Messages path")

    with rebound(dispatch, "_PYTHON_MESSAGES", python), rebound(dispatch, "_PYTHON_AMESSAGES", python):
        result: Final = await _public(asynchronous, "anthropic/claude-sonnet-4-5", kwargs)

    assert isinstance(result, dict)
    assert shipped_native == ["async" if asynchronous else "sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize(
    "callbacks",
    (
        pytest.param([_Agentic()], id="agentic-loop"),
        pytest.param([WebSearchInterceptionLogger(enabled_providers=["anthropic"])], id="websearch-interception"),
    ),
)
async def test_anthropic_serves_agentic_loop_callbacks_on_rust(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool, callbacks: list[object]
) -> None:
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("anthropic has no Python Messages path")

    with rebound(dispatch, "_PYTHON_MESSAGES", python), rebound(dispatch, "_PYTHON_AMESSAGES", python):
        result: Final = await _public(asynchronous, "anthropic/claude-sonnet-4-5", {})

    assert isinstance(result, dict)
    assert shipped_native == ["async" if asynchronous else "sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize(
    "server_url",
    (
        pytest.param("litellm_proxy", id="whole-gateway"),
        pytest.param("litellm_proxy/mcp/deepwiki", id="one-server"),
        pytest.param("https://gateway.example/mcp/deepwiki", id="proxy-path"),
    ),
)
async def test_anthropic_serves_gateway_mcp_tools_on_rust(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool, server_url: str
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    tools: Final = [{"type": "mcp", "server_label": "gateway", "server_url": server_url, "require_approval": "never"}]

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("anthropic has no Python Messages path")

    with rebound(dispatch, "_PYTHON_MESSAGES", python), rebound(dispatch, "_PYTHON_AMESSAGES", python):
        result: Final = await _public(asynchronous, "anthropic/claude-sonnet-4-5", {"tools": tools})

    assert isinstance(result, dict)
    assert shipped_native == ["async" if asynchronous else "sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
async def test_other_providers_keep_python_for_the_same_features(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    python_result: Final = response("deepseek/deepseek-v4-flash")

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        return python_result

    async def apython(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        return python_result

    with rebound(dispatch, "_PYTHON_MESSAGES", python), rebound(dispatch, "_PYTHON_AMESSAGES", apython):
        result: Final = await _public(asynchronous, "deepseek/deepseek-v4-flash", {"client": object()})

    assert result is python_result
    assert shipped_native == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "refused"),
    (
        pytest.param("anthropic/claude-sonnet-4-5", True, id="anthropic"),
        pytest.param("deepseek/deepseek-v4-flash", False, id="deepseek"),
    ),
)
async def test_a_failing_detector_refuses_only_the_rust_only_provider(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], model: str, refused: bool
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    python_result: Final = response(model)

    def failing(request: NativeCall) -> bool:
        raise RuntimeError("detector failed")

    async def apython(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        return python_result

    with (
        rebound(dispatch, "_FEATURES", {Feature.MOCK_RESPONSE: failing}),
        rebound(dispatch, "_PYTHON_AMESSAGES", apython),
    ):
        if not refused:
            assert await _public(True, model, {}) is python_result
            return
        with pytest.raises(UnsupportedParamsError, match="does not support this call"):
            await _public(True, model, {})
    assert shipped_native == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
async def test_an_anthropic_call_its_signature_rejects_keeps_python_s_type_error(
    monkeypatch: pytest.MonkeyPatch, shipped_native: list[str], asynchronous: bool
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    call: Final = {"messages": MESSAGES, "model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"}

    async def attempt() -> object:
        if asynchronous:
            return await dispatch.anthropic_messages(**call)
        return dispatch.anthropic_messages_handler(**call)

    with pytest.raises(TypeError, match="max_tokens"):
        await attempt()

    assert shipped_native == []
