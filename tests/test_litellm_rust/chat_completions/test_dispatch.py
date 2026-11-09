from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.chat_completions import dispatch
from litellm.chat_completions.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Feature, Route, RouteRule, Rules
from litellm.rust_bridge.chat_completions.entrypoints import (
    NATIVE_ACOMPLETION,
    NATIVE_COMPLETION,
    NativeAcompletion,
    NativeCompletion,
)
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.public_call import NativeCall, native_call_hook
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]
PYTHON_RULES: Final = ()
RUST_RULES: Final = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)


def completion_binding(native: NativeCompletion | None) -> NativeBinding[NativeCompletion]:
    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    return binding


def acompletion_binding(native: NativeAcompletion | None) -> NativeBinding[NativeAcompletion]:
    binding: Final[NativeBinding[NativeAcompletion]] = NativeBinding("acompletion", validate=lambda _: None)
    binding.override(native)
    return binding


def test_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = ("gpt-4o", MESSAGES)
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = ModelResponse()

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records public call shape
        captured.append((call_args, call_kwargs))
        return response

    def native(request: NativeCall) -> ModelResponse:
        pytest.fail("Python-only dispatch must not call native")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=completion_binding(native),
            native=native_call_hook,
            rules=PYTHON_RULES,
        )
        is response
    )
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "metadata": metadata}


@pytest.mark.asyncio
async def test_async_python_route_forwards_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    args: Final[tuple[object, ...]] = ("gpt-4o", MESSAGES)
    kwargs: Final[Mapping[str, object]] = {"temperature": 0.1, "metadata": metadata}
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = ModelResponse()

    async def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records call shape
        captured.append((call_args, call_kwargs))
        return response

    async def native(request: NativeCall) -> ModelResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=acompletion_binding(native),
        native=native_call_hook,
        rules=PYTHON_RULES,
    )
    assert result is response
    call_args, call_kwargs = captured[0]
    assert call_args == args
    assert call_args[1] is MESSAGES
    assert call_kwargs == kwargs
    assert call_kwargs["metadata"] is metadata
    assert kwargs == {"temperature": 0.1, "metadata": metadata}


def test_native_receives_bound_request_and_original_call_shape() -> None:
    metadata: Final = {"user_id": "u"}
    kwargs: Final[Mapping[str, object]] = {
        "stream": True,
        "api_key": "sk-test",
        "base_url": "https://example.invalid",
        "extra_headers": {"x-test": "1"},
        "custom_llm_provider": "anthropic",
        "metadata": metadata,
    }
    captured: Final[list[tuple[NativeCall, tuple[object, ...], Mapping[str, object]]]] = []

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: rejected Rust fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(request: NativeCall) -> ModelResponse:
        captured.append((request, request.args, request.kwargs))
        return ModelResponse()

    args: Final[tuple[object, ...]] = ("anthropic/claude-sonnet-4-5", MESSAGES)
    _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=completion_binding(native),
        native=native_call_hook,
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert request.resolved["model"] == "anthropic/claude-sonnet-4-5"
    assert request.resolved["messages"] is MESSAGES
    assert request.resolved["stream"] is True
    assert request.resolved["api_key"] == "sk-test"
    assert request.resolved["base_url"] == "https://example.invalid"
    assert request.resolved["custom_llm_provider"] == "anthropic"
    assert request.resolved["extra_headers"] == {"x-test": "1"}
    assert request.kwargs is kwargs
    assert call_args == args
    assert call_kwargs == kwargs
    assert call_kwargs["metadata"] is metadata


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        (("gpt-4o", MESSAGES), {"model": "duplicate"}),
        ((), {}),
    ),
)
def test_binding_errors_delegate_to_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    captured: Final[list[tuple[tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = ModelResponse()

    def python(*call_args: object, **call_kwargs: object) -> ModelResponse:  # kwargs-ok: records invalid call shape
        captured.append((call_args, call_kwargs))
        return response

    def native(request: NativeCall) -> ModelResponse:
        pytest.fail("Binding failures must be delegated to Python")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=completion_binding(native),
            native=native_call_hook,
            rules=RUST_RULES,
        )
        is response
    )
    assert captured == [(args, kwargs)]


def test_sync_completion_request_projects_public_arguments() -> None:
    rules: Final[Rules] = (RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),)
    expected: Final = ModelResponse()

    def native(request: NativeCall) -> ModelResponse:
        assert request.resolved["model"] == "test-model"
        assert request.resolved["messages"] == MESSAGES
        assert request.resolved["custom_llm_provider"] == "openai"
        assert request.resolved["stream"] is True
        return expected

    binding: Final[NativeBinding[NativeCompletion]] = NativeBinding("completion", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", MESSAGES),
        {"custom_llm_provider": "openai", "stream": True},
        python=lambda *args, **kwargs: pytest.fail("required native route must handle this call"),
        binding=binding,
        native=native_call_hook,
        rules=rules,
    )

    assert response is expected


def test_positional_parameters_remain_available_to_native_projection() -> None:
    request: Final = _DISPATCH.request(("anthropic/test-model", MESSAGES, 12.0, 0.25), {})
    assert request is not None
    assert request.resolved["timeout"] == 12.0
    assert request.resolved["temperature"] == 0.25
    assert request.resolved["messages"] is MESSAGES


@pytest.mark.asyncio
async def test_acompletion_marker_binds_positionals_by_the_sync_signature() -> None:
    captured: Final[list[NativeCall]] = []

    async def native(call: NativeCall) -> ModelResponse:
        captured.append(call)
        return ModelResponse()

    NATIVE_ACOMPLETION.override(native)
    public_completion: Final = cast(Callable[..., Awaitable[ModelResponse]], litellm.completion)
    try:
        _ = await public_completion("gpt-4o", MESSAGES, 30.0, acompletion=True)
    finally:
        NATIVE_ACOMPLETION.reset()
    assert [(request.resolved["timeout"], request.resolved["functions"]) for request in captured] == [(30.0, None)]


class _ChatAgentic(CustomLogger):
    async def async_should_run_chat_completion_agentic_loop(
        self, *args: object, **kwargs: object
    ) -> tuple[bool, dict[str, object]]:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double only needs the override
        return False, {}


@pytest.mark.parametrize(
    ("feature", "kwargs", "settings"),
    (
        (Feature.MOCK_RESPONSE, {"mock_response": "hi"}, {}),
        pytest.param(Feature.MOCK_RESPONSE, {"mock_tool_calls": [{"id": "call_1"}]}, {}, id="mock-tool-calls"),
        pytest.param(Feature.MOCK_RESPONSE, {"mock_timeout": True}, {}, id="mock-timeout"),
        (Feature.CALLER_CLIENT, {"client": object()}, {}),
        (Feature.TIMEOUT_VARIANTS, {"request_timeout": 5}, {}),
        (Feature.AGENTIC_LOOP_HOOK, {}, {"callbacks": [_ChatAgentic()]}),
        (Feature.TLS_POLICY, {}, {"ssl_security_level": "DEFAULT@SECLEVEL=1"}),
        (Feature.RESPONSE_CACHE, {}, {"cache": Cache()}),
        (Feature.STREAMING, {"stream": True}, {}),
        (Feature.DROP_PARAMS, {}, {"drop_params": True}),
        (Feature.MODIFY_PARAMS, {}, {"modify_params": True}),
        (Feature.EXTRA_BODY, {"extra_body": {"top_k": 1}}, {}),
        (Feature.RETRIES_FALLBACKS, {"num_retries": 2}, {}),
        pytest.param(
            Feature.RETRIES_FALLBACKS,
            {"context_window_fallback_dict": {"anthropic/claude-sonnet-4-5": "anthropic/claude-opus-4-5"}},
            {},
            id="context-window-fallbacks",
        ),
        pytest.param(
            Feature.RETRIES_FALLBACKS,
            {},
            {"model_fallbacks": [{"anthropic/claude-sonnet-4-5": ["anthropic/claude-opus-4-5"]}]},
            id="model-fallbacks",
        ),
        (Feature.PROMPT_MANAGEMENT, {"prompt_id": "welcome"}, {}),
        pytest.param(Feature.PROMPT_MANAGEMENT, {"model": "litellm_agent/gpt-5"}, {}, id="litellm-agent-model"),
        (Feature.CACHE_CONTROL_INJECTION, {"enable_prompt_caching": True}, {}),
        (Feature.MODEL_PARAM_GATING, {"logit_bias": {"1": 1}}, {}),
        (
            Feature.MODEL_LIST,
            {"model_list": [{"model_name": "anthropic/claude-sonnet-4-5", "litellm_params": {"model": "gpt-5"}}]},
            {},
        ),
        (
            Feature.MODEL_ALIAS_MAP,
            {},
            {"model_alias_map": {"anthropic/claude-sonnet-4-5": "anthropic/claude-opus-4-5"}},
        ),
    ),
)
def test_each_detected_feature_reaches_its_rule(
    monkeypatch: pytest.MonkeyPatch, feature: Feature, kwargs: Mapping[str, object], settings: Mapping[str, object]
) -> None:
    rules: Final = (
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY, features=frozenset({feature})),
        RouteRule(Route.CHAT_COMPLETIONS, Rollout.RUST_REQUIRED),
    )
    python_result: Final = ModelResponse()
    native_result: Final = ModelResponse()

    def served(call_kwargs: Mapping[str, object]) -> object:
        return _DISPATCH.run(
            (),
            {"model": "anthropic/claude-sonnet-4-5", "messages": MESSAGES, **call_kwargs},
            python=lambda *call_args, **python_kwargs: python_result,
            binding=completion_binding(lambda request: native_result),
            native=native_call_hook,
            rules=rules,
        )

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "ssl_security_level", None)
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "model_alias_map", {})
    for name in ("num_retries", "model_fallbacks"):
        monkeypatch.setattr(litellm, name, None)
    for name in ("drop_params", "modify_params", "enable_anthropic_prompt_caching"):
        monkeypatch.setattr(litellm, name, False)
    for name in ("SSL_SECURITY_LEVEL", "SSL_ECDH_CURVE"):
        monkeypatch.delenv(name, raising=False)
    assert served({}) is native_result
    for name, value in settings.items():
        monkeypatch.setattr(litellm, name, value)
    assert served(kwargs) is python_result


def test_acompletion_flag_hands_off_only_when_true() -> None:
    expected: Final = ModelResponse()

    def native(call: NativeCall) -> ModelResponse:
        return expected

    async def anative(call: NativeCall) -> ModelResponse:
        pytest.fail("completion(..., acompletion=1) stays sync, since the Python @client only hands off on is True")

    NATIVE_COMPLETION.override(native)
    NATIVE_ACOMPLETION.override(anative)
    public_completion: Final = cast(Callable[..., object], litellm.completion)
    try:
        result: Final = public_completion("gpt-4o", MESSAGES, acompletion=1)
    finally:
        NATIVE_COMPLETION.reset()
        NATIVE_ACOMPLETION.reset()
    assert result is expected
