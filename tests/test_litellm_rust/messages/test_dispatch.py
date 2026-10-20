import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.llms.anthropic.pass_through.messages import handler as python_messages
from litellm.messages import dispatch
from litellm.messages.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.proxy import proxy_server
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Feature, Route, RouteRule, Rules, Target
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES, NativeAmessages, NativeMessages
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

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


@pytest.mark.parametrize("stream, expected_timeout", ((True, 0.5), (False, 30.0)))
def test_native_streams_are_bounded_by_stream_timeout(stream: bool, expected_timeout: float) -> None:
    captured: Final[list[NativeCall]] = []

    def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(request: NativeCall) -> AnthropicMessagesResponse:
        captured.append(request)
        return response()

    _ = _DISPATCH.run(
        (16, MESSAGES, "anthropic/claude-sonnet-4-5"),
        {"stream": stream, "timeout": 30.0, "stream_timeout": 0.5},
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
    ("model", "kwargs", "general_settings", "serves_python"),
    (
        ("deepseek/deepseek-v4-flash", {"safeguards": DANGEROUS_TOOL_USE}, CLASSIFIER_SETTINGS, True),
        (
            "deepseek-v4-flash",
            {"safeguards": DANGEROUS_TOOL_USE, "custom_llm_provider": "deepseek"},
            CLASSIFIER_SETTINGS,
            True,
        ),
        ("deepseek/deepseek-v4-flash", {"safeguards": DANGEROUS_TOOL_USE}, {}, False),
        ("deepseek/deepseek-v4-flash", {}, CLASSIFIER_SETTINGS, False),
        ("deepseek/deepseek-v4-flash", {"safeguards": [{"type": "other"}]}, CLASSIFIER_SETTINGS, False),
        ("anthropic/claude-sonnet-4-5", {"safeguards": DANGEROUS_TOOL_USE}, CLASSIFIER_SETTINGS, False),
    ),
)
def test_proxy_classified_safeguards_reach_the_feature_rule(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    kwargs: Mapping[str, object],
    general_settings: Mapping[str, object],
    serves_python: bool,
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", general_settings)
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
        rules=(
            RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({Feature.SAFEGUARDS_CLASSIFIER})),
            RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
        ),
    )
    assert result is (python_result if serves_python else native_result)


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


ADVISOR_TOOL: Final = {"type": ANTHROPIC_ADVISOR_TOOL_TYPE, "name": "advisor", "model": "claude-opus-5-5"}


@pytest.mark.parametrize(
    ("feature", "model", "kwargs", "settings"),
    (
        (Feature.MOCK_RESPONSE, "anthropic/claude-sonnet-4-5", {"mock_response": "hi"}, {}),
        (
            Feature.PROXY_MCP_TOOLS,
            "anthropic/claude-sonnet-4-5",
            {"tools": [{"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}]},
            {},
        ),
        (Feature.CALLER_CLIENT, "anthropic/claude-sonnet-4-5", {"client": object()}, {}),
        (Feature.TIMEOUT_VARIANTS, "anthropic/claude-sonnet-4-5", {"request_timeout": 5}, {}),
        (Feature.AGENTIC_LOOP_HOOK, "anthropic/claude-sonnet-4-5", {}, {"callbacks": [_Agentic()]}),
        (Feature.RESPONSE_CACHE, "anthropic/claude-sonnet-4-5", {}, {"cache": Cache()}),
        (Feature.TLS_POLICY, "anthropic/claude-sonnet-4-5", {}, {"ssl_security_level": "DEFAULT@SECLEVEL=1"}),
        (Feature.MODIFY_PARAMS, "anthropic/claude-sonnet-4-5", {}, {"modify_params": True}),
        (
            Feature.WORKLOAD_IDENTITY,
            "anthropic/claude-sonnet-4-5",
            {"anthropic_federation_rule_id": "rule-1", "anthropic_organization_id": "org-1"},
            {},
        ),
        (Feature.CACHE_CONTROL_INJECTION, "anthropic/claude-sonnet-4-5", {"enable_prompt_caching": True}, {}),
        (
            Feature.THINKING_SIGNATURE_RETRY,
            "anthropic/claude-sonnet-4-5",
            {"messages": [{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]}]},
            {},
        ),
        (
            Feature.STREAM_COST_INJECTION,
            "anthropic/claude-sonnet-4-5",
            {"stream": True},
            {"include_cost_in_streaming_usage": True},
        ),
    ),
)
def test_each_detected_feature_reaches_its_rule(
    monkeypatch: pytest.MonkeyPatch,
    feature: Feature,
    model: str,
    kwargs: Mapping[str, object],
    settings: Mapping[str, object],
) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({feature})),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    python_result: Final = response(model)
    native_result: Final = response(model)

    def served(call_kwargs: Mapping[str, object]) -> AnthropicMessagesResponse:
        messages: Final = call_kwargs.get("messages", MESSAGES)
        return _DISPATCH.run(
            (16, messages, model),
            {name: value for name, value in call_kwargs.items() if name != "messages"},
            python=lambda *call_args, **python_kwargs: python_result,
            binding=messages_binding(lambda request: native_result),
            native=lambda hook, request, call_args, native_kwargs: hook(request),
            rules=rules,
        )

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "ssl_security_level", None)
    for setting in ("modify_params", "enable_anthropic_prompt_caching", "include_cost_in_streaming_usage"):
        monkeypatch.setattr(litellm, setting, False)
    for name in ("SSL_SECURITY_LEVEL", "SSL_ECDH_CURVE", "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert served({}) is native_result
    for name, value in settings.items():
        monkeypatch.setattr(litellm, name, value)
    assert served(kwargs) is python_result


ASYNC_ONLY_FEATURES: Final = (
    pytest.param(Feature.PRE_REQUEST_HOOK, {}, {"callbacks": [_Rewrite()]}, id="pre-request-hook"),
    pytest.param(
        Feature.WEBSEARCH_INTERCEPTION, {}, {"callbacks": [WebSearchInterceptionLogger()]}, id="websearch-interception"
    ),
    pytest.param(
        Feature.MESSAGES_INTERCEPTOR,
        {"model": "deepseek/deepseek-v4-flash", "tools": [ADVISOR_TOOL]},
        {},
        id="messages-interceptor",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("feature", "kwargs", "settings"), ASYNC_ONLY_FEATURES)
async def test_handler_hooks_the_sync_entry_never_runs_keep_only_async_calls_on_python(
    monkeypatch: pytest.MonkeyPatch, feature: Feature, kwargs: Mapping[str, object], settings: Mapping[str, object]
) -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({feature})),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    python_result: Final = response()
    native_result: Final = response()
    model: Final = str(kwargs.get("model", "anthropic/claude-sonnet-4-5"))
    call_kwargs: Final = {name: value for name, value in kwargs.items() if name != "model"}

    async def python(*call_args: object, **python_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        return python_result

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        return native_result

    monkeypatch.setattr(litellm, "callbacks", [])
    for name, value in settings.items():
        monkeypatch.setattr(litellm, name, value)
    served_async: Final = await _ADISPATCH.arun(
        (16, MESSAGES, model),
        call_kwargs,
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, args, native_kwargs: hook(request),
        rules=rules,
    )
    served_sync: Final = _DISPATCH.run(
        (16, MESSAGES, model),
        call_kwargs,
        python=lambda *call_args, **python_kwargs: python_result,
        binding=messages_binding(lambda request: native_result),
        native=lambda hook, request, args, native_kwargs: hook(request),
        rules=rules,
    )
    assert served_async is python_result
    assert served_sync is native_result


@pytest.mark.asyncio
async def test_interceptor_feature_follows_the_resolved_provider() -> None:
    rules: Final[Rules] = (
        RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=frozenset({Feature.MESSAGES_INTERCEPTOR})),
        RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),
    )
    native_result: Final = response()

    async def python(*call_args: object, **call_kwargs: object) -> AnthropicMessagesResponse:  # kwargs-ok: stub
        pytest.fail("anthropic serves the advisor tool itself")

    async def native(request: NativeCall) -> AnthropicMessagesResponse:
        return native_result

    result: Final = await _ADISPATCH.arun(
        (16, MESSAGES, "claude-sonnet-4-5"),
        {"tools": [ADVISOR_TOOL], "custom_llm_provider": "anthropic"},
        python=python,
        binding=amessages_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=rules,
    )
    assert result is native_result
