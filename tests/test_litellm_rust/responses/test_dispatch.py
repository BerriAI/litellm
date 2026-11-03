import inspect
from collections.abc import Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures for inspect

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.responses import dispatch as responses_dispatch
from litellm.responses import main as python_responses
from litellm.responses.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Feature, Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.responses.entrypoints import (
    NativeAresponses,
    NativeResponses,
)
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES
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
        request: NativeCall,
    ) -> ResponsesAPIResponse:
        pytest.fail("Python-only dispatch must not call native")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=responses_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request),
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
        request: NativeCall,
    ) -> ResponsesAPIResponse:
        pytest.fail("Python-only dispatch must not call native")

    result: Final = await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=aresponses_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
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
    captured: Final[list[tuple[NativeCall, tuple[object, ...], Mapping[str, object]]]] = []
    response: Final = _response("anthropic/claude-sonnet-4-5")

    def python(*call_args: object, **call_kwargs: object) -> ResponsesAPIResponse:  # kwargs-ok: rejected fallback
        pytest.fail("Required Rust dispatch must not call Python")

    def native(
        request: NativeCall,
    ) -> ResponsesAPIResponse:
        args: Final = request.args
        kwargs: Final = request.kwargs
        captured.append((request, args, kwargs))
        return response

    result: Final = _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=responses_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=RUST_RULES,
    )

    request, call_args, call_kwargs = captured[0]
    assert result is response
    assert request.resolved["model"] == "anthropic/claude-sonnet-4-5"
    assert request.resolved["input"] is INPUT
    assert request.resolved["stream"] is True
    assert request.resolved["api_key"] == "sk-test"
    assert request.resolved["base_url"] == "https://example.invalid"
    assert request.resolved["custom_llm_provider"] == "anthropic"
    assert request.resolved["extra_headers"] is extra_headers
    assert request.kwargs == kwargs
    assert request.kwargs["litellm_metadata"] is metadata
    assert call_args == args
    assert call_args[0] is INPUT
    assert call_kwargs == kwargs
    assert call_kwargs["extra_headers"] is extra_headers
    assert call_kwargs["litellm_metadata"] is metadata


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
        request: NativeCall,
    ) -> ResponsesAPIResponse:
        pytest.fail("Binding failures must be delegated to Python")

    assert (
        _DISPATCH.run(
            args,
            kwargs,
            python=python,
            binding=responses_binding(native),
            native=lambda hook, request, call_args, call_kwargs: hook(request),
            rules=RUST_RULES,
        )
        is response
    )
    assert captured == [(args, kwargs)]


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


def test_positional_parameters_remain_available_to_native_projection() -> None:
    include: Final = ["reasoning.encrypted_content"]
    request: Final = _DISPATCH.request((INPUT, "openai/test-model", include, "Be brief", 16), {})
    assert request is not None
    assert request.resolved["include"] is include
    assert request.resolved["instructions"] == "Be brief"
    assert request.resolved["max_output_tokens"] == 16


class _Agentic(CustomLogger):
    async def async_should_run_agentic_loop(self, *args: object, **kwargs: object) -> tuple[bool, dict]:  # pyright: ignore[reportIncompatibleMethodOverride, reportMissingTypeArgument]  # test double only needs the override
        return False, {}


class _Deployment(CustomLogger):
    async def async_pre_call_deployment_hook(self, *args: object, **kwargs: object) -> dict[str, object] | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double only needs the override
        return None


ENCODED_ID: Final = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-1", "rs_1")  # pyright: ignore[reportPrivateUsage]  # the encoder litellm stamps on earlier turns


@pytest.mark.parametrize(
    ("feature", "kwargs", "settings"),
    (
        (Feature.MOCK_RESPONSE, {"mock_response": "hi"}, {}),
        (Feature.PROXY_MCP_TOOLS, {"tools": [{"type": "mcp", "server_label": "g", "server_url": "litellm_proxy"}]}, {}),
        (Feature.CALLER_CLIENT, {"client": object()}, {}),
        pytest.param(Feature.CALLER_CLIENT, {"shared_session": object()}, {}, id="shared-session"),
        pytest.param(Feature.EXTRA_BODY, {"extra_body": {"stream": True}}, {}, id="extra-body-stream"),
        pytest.param(
            Feature.PROVIDER_AFFINITY,
            {"provider_affinity_header": "X-Session-ID", "session_id": "session-1"},
            {},
            id="provider-affinity",
        ),
        pytest.param(
            Feature.INPUT_REWRITE,
            {"input": [{"type": "reasoning", "id": "rs_1", "summary": list[object](), "status": None}]},
            {},
            id="input-rewrite",
        ),
        (Feature.TIMEOUT_VARIANTS, {"request_timeout": 5}, {}),
        (Feature.AGENTIC_LOOP_HOOK, {}, {"callbacks": [_Agentic()]}),
        (Feature.TLS_POLICY, {}, {"ssl_security_level": "DEFAULT@SECLEVEL=1"}),
        (Feature.DROP_PARAMS, {}, {"drop_params": True}),
        (Feature.STREAMING, {"stream": True}, {}),
        (Feature.PROMPT_MANAGEMENT, {"prompt_id": "welcome"}, {}),
        (Feature.TEXT_FORMAT, {"text_format": dict}, {}),
        (Feature.CHAT_COMPLETIONS_BRIDGE, {"use_chat_completions_api": True}, {}),
        (Feature.CHAT_COMPLETIONS_BRIDGE, {"model": "openai/chat_completions/gpt-5"}, {}),
        (Feature.RETRIES_FALLBACKS, {"num_retries": 2}, {}),
        (Feature.DEPLOYMENT_RESPONSE_ID, {"litellm_metadata": {"model_info": {"id": "deployment-1"}}}, {}),
        (Feature.ENCODED_ITEM_IDS, {"input": [{"type": "reasoning", "id": ENCODED_ID, "summary": []}]}, {}),
        (Feature.MODIFY_PARAMS, {}, {"modify_params": True}),
        (Feature.RESPONSE_CACHE, {}, {"cache": Cache()}),
        (
            Feature.MANAGED_RESOURCE_IDS,
            {
                "input": [{"role": "user", "content": [{"type": "input_file", "file_id": "litellm-file"}]}],
                "model_file_id_mapping": {"litellm-file": {"deployment-1": "file-abc"}},
                "model_info": {"id": "deployment-1"},
            },
            {},
        ),
        (Feature.CACHE_CONTROL_MARKS, {"tools": [{"type": "function", "name": "f", "cache_control": {}}]}, {}),
        (
            Feature.ANTHROPIC_TOOL_SEARCH,
            {"tools": [{"type": sorted(ANTHROPIC_TOOL_SEARCH_TOOL_TYPES)[0], "name": "s"}]},
            {},
        ),
        (Feature.TOOL_SCHEMA_REPAIR, {"tools": [{"type": "function", "name": "f", "parameters": "{}"}]}, {}),
        (Feature.MODEL_PARAM_GATING, {"max_output_tokens": 1}, {}),
        (Feature.CLIENT_HEADERS, {"headers": {"x-tenant": "a"}}, {}),
        (Feature.DEPLOYMENT_HOOK, {}, {"callbacks": [_Deployment()]}),
        (
            Feature.WORKLOAD_IDENTITY,
            {
                "openai_identity_provider_id": "idp-1",
                "openai_service_account_id": "sa-1",
                "openai_identity_token_file": "/var/run/token",
            },
            {},
        ),
    ),
)
def test_each_detected_feature_reaches_its_rule(
    monkeypatch: pytest.MonkeyPatch, feature: Feature, kwargs: Mapping[str, object], settings: Mapping[str, object]
) -> None:
    rules: Final = (
        RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY, features=frozenset({feature})),
        RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),
    )
    python_result: Final = _response()
    native_result: Final = _response()

    def served(call_kwargs: Mapping[str, object]) -> object:
        return _DISPATCH.run(
            (),
            {"model": "gpt-5", "input": INPUT, **call_kwargs},
            python=lambda *call_args, **python_kwargs: python_result,
            binding=responses_binding(lambda request: native_result),
            native=lambda hook, request, call_args, native_kwargs: hook(request),
            rules=rules,
        )

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "modify_params", False)
    monkeypatch.setattr(litellm, "ssl_security_level", None)
    monkeypatch.setattr(litellm, "cache", None)
    for setting in ("api_key", "openai_key", "api_base", "num_retries"):
        monkeypatch.setattr(litellm, setting, None)
    for name in ("SSL_SECURITY_LEVEL", "SSL_ECDH_CURVE", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert served({}) is native_result
    for name, value in settings.items():
        monkeypatch.setattr(litellm, name, value)
    assert served(kwargs) is python_result


@pytest.mark.asyncio
async def test_deployment_hook_keeps_only_sync_calls_on_python(monkeypatch: pytest.MonkeyPatch) -> None:
    rules: Final = (
        RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY, features=frozenset({Feature.DEPLOYMENT_HOOK})),
        RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED),
    )
    native_result: Final = _response()

    async def python(*call_args: object, **call_kwargs: object) -> ResponsesAPIResponse:  # kwargs-ok: stub
        pytest.fail("the async handler leaves the deployment hook to the shared call wrapper")

    async def native(request: NativeCall) -> ResponsesAPIResponse:
        return native_result

    monkeypatch.setattr(litellm, "callbacks", [_Deployment()])
    result: Final = await _ADISPATCH.arun(
        (),
        {"model": "gpt-5", "input": INPUT},
        python=python,
        binding=aresponses_binding(native),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=rules,
    )
    assert result is native_result


@pytest.mark.parametrize(
    ("kwargs", "native"),
    (
        ({"model": "gpt-5"}, True),
        ({"model": "openai/gpt-5"}, True),
        ({"model": "gpt-5", "custom_llm_provider": "openai"}, True),
        ({"model": "anthropic/claude-sonnet-4-5"}, False),
        ({"model": "claude-sonnet-4-5"}, False),
        ({"model": "gpt-5", "custom_llm_provider": "azure"}, False),
    ),
)
def test_provider_rule_sees_the_resolved_provider(kwargs: Mapping[str, object], native: bool) -> None:
    python_result: Final = _response()
    native_result: Final = _response()

    result: Final = _DISPATCH.run(
        (),
        {"input": INPUT, **kwargs},
        python=lambda *call_args, **call_kwargs: python_result,
        binding=responses_binding(lambda request: native_result),
        native=lambda hook, request, call_args, call_kwargs: hook(request),
        rules=(RouteRule(Route.RESPONSES, Rollout.RUST_REQUIRED, providers=frozenset({"openai"})),),
    )
    assert result is (native_result if native else python_result)
