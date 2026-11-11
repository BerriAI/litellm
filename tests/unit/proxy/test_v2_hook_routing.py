import asyncio
from collections.abc import Generator, Sequence
from typing import Final

import pytest
from fastapi import HTTPException
from mcp.types import CallToolResult, TextContent

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Field, Hook, build_manifest, on
from litellm.integrations.custom_logger_v2.host import register_native, reset_host_for_tests
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral, GenericGuardrailAPIInputs, ModelResponse

AUTH: Final = UserAPIKeyAuth(api_key="sk-test")


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    yield
    reset_host_for_tests()


@pytest.fixture
def proxy_logging() -> ProxyLogging:
    return ProxyLogging(user_api_key_cache=DualCache())


class _Witness(CustomLoggerV2):
    def __init__(self, order: list[str]) -> None:
        self.order = order

    @on(Hook.pre_call, may_abort=True)
    async def pre_call(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.during_call, may_abort=True)
    async def during_call(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.post_call_success, may_abort=True)
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.post_call_failure, may_abort=True)
    async def post_call_failure(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.response_headers, writes={Field.response_headers})
    async def response_headers(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.list_models, may_abort=True)
    async def list_models(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")

    @on(Hook.mcp_tool_result, may_abort=True)
    async def mcp_tool_result(self, view: object, ctx: CallContext[object]) -> None:
        self.order.append("native")


class _Legacy(CustomLogger):
    def __init__(self, order: list[str], reply: object = None, error: Exception | None = None) -> None:
        super().__init__()
        self.order = order
        self.reply = reply
        self.error = error
        self.seen: list[object] = []

    def _record(self, seen: object) -> object:
        self.order.append("legacy")
        self.seen.append(seen)
        if self.error is not None:
            raise self.error
        return self.reply

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> object:
        return self._record(data)

    async def async_moderation_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> object:
        return self._record(data)

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        return self._record(response)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> object:
        return self._record(original_exception)

    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> object:
        return self._record(response)

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        self._record(tuple(model_names))
        return ("kept",)


class _McpGuardrail(CustomGuardrail):
    def __init__(self, order: list[str]) -> None:
        super().__init__(guardrail_name="mcp-witness", default_on=True, event_hook=GuardrailEventHooks.post_mcp_call)
        self.order = order

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: str,
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        self.order.append("legacy")
        return inputs


class _StreamGuardrail(CustomGuardrail):
    def __init__(self, order: list[str], reply: object) -> None:
        super().__init__(guardrail_name="stream-witness", default_on=True, event_hook=GuardrailEventHooks.post_call)
        self.order = order
        self.reply = reply

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.order.append("legacy")
        return self.reply


class _SuccessDispatch:
    def __init__(self) -> None:
        self.model_call_details: dict[str, object] = {"metadata": {}}
        self.dispatched: list[object] = []

    async def dispatch_success_handlers(self, result: object, **kwargs: object) -> None:
        self.dispatched.append(result)


async def test_pre_call_passthrough_hands_the_request_to_legacy_once_and_returns_its_reply(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    reply: Final[dict[str, object]] = {"model": "replaced", "metadata": {}}
    legacy: Final = _Legacy(order, reply=reply)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    data: Final[dict[str, object]] = {"model": "gpt-4o", "metadata": {}}

    result: Final = await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert result is reply
    assert legacy.seen == [data]
    assert legacy.seen[0] is data


async def test_pre_call_passthrough_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = ValueError("rejected by a legacy hook")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])

    with pytest.raises(ValueError) as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "gpt-4o"}, call_type="completion")

    assert raised.value is error


async def test_pre_call_native_runs_before_legacy(monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "gpt-4o"}, call_type="completion")

    assert order == ["native", "legacy"]


async def test_during_call_passthrough_returns_the_request_and_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    legacy: Final = _Legacy(order)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    data: Final = {"model": "gpt-4o"}

    result: Final = await proxy_logging.during_call_hook(data=data, user_api_key_dict=AUTH, call_type="completion")

    assert result is data
    assert legacy.seen == [data]
    error: Final = RuntimeError("moderation failed")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])
    with pytest.raises(RuntimeError) as raised:
        await proxy_logging.during_call_hook(data=data, user_api_key_dict=AUTH, call_type="completion")
    assert raised.value is error


async def test_during_call_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.during_call_hook(data={"model": "gpt-4o"}, user_api_key_dict=AUTH, call_type="completion")

    assert order == ["native", "legacy"]


async def test_post_call_success_passthrough_returns_the_legacy_replacement(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    replacement: Final = ModelResponse()
    legacy: Final = _Legacy([], reply=replacement)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    response: Final = ModelResponse()

    result: Final = await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response=response, user_api_key_dict=AUTH
    )

    assert result is replacement
    assert legacy.seen == [response]
    assert legacy.seen[0] is response


async def test_post_call_success_passthrough_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = ValueError("response rejected")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])

    with pytest.raises(ValueError) as raised:
        await proxy_logging.post_call_success_hook(
            data={"model": "gpt-4o"}, response=ModelResponse(), user_api_key_dict=AUTH
        )

    assert raised.value is error


async def test_post_call_success_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response=ModelResponse(), user_api_key_dict=AUTH
    )

    assert order == ["native", "legacy"]


async def test_post_call_failure_passthrough_returns_the_first_legacy_http_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    transformed: Final = HTTPException(status_code=418, detail="rewritten")
    legacy: Final = _Legacy([], reply=transformed)
    monkeypatch.setattr(litellm, "callbacks", [legacy, _Legacy([], reply=HTTPException(status_code=500))])
    original: Final = RuntimeError("upstream failed")

    result: Final = await proxy_logging.post_call_failure_hook(
        request_data={"model": "gpt-4o"}, original_exception=original, user_api_key_dict=AUTH
    )

    assert result is transformed
    assert legacy.seen == [original]
    assert legacy.seen[0] is original


async def test_post_call_failure_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "gpt-4o"}, original_exception=RuntimeError("failed"), user_api_key_dict=AUTH
    )

    assert order == ["native", "legacy"]


async def test_response_headers_passthrough_returns_the_legacy_headers(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    legacy: Final = _Legacy([], reply={"x-legacy": "1"})
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    response: Final = ModelResponse()

    result: Final = await proxy_logging.post_call_response_headers_hook(
        data={"model": "gpt-4o"}, user_api_key_dict=AUTH, response=response
    )

    assert result == {"x-legacy": "1"}
    assert legacy.seen == [response]


async def test_response_headers_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order, reply={"x-legacy": "1"})])

    await proxy_logging.post_call_response_headers_hook(
        data={"model": "gpt-4o"}, user_api_key_dict=AUTH, response=ModelResponse()
    )

    assert order == ["native", "legacy"]


async def test_list_models_passthrough_returns_the_names_legacy_hides(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    legacy: Final = _Legacy([])
    monkeypatch.setattr(litellm, "callbacks", [legacy])

    result: Final = await proxy_logging.hidden_by_listing_callbacks(AUTH, ["kept", "hidden"])

    assert result == frozenset({"hidden"})
    assert legacy.seen == [("kept", "hidden")]


async def test_list_models_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_Legacy(order)])

    await proxy_logging.hidden_by_listing_callbacks(AUTH, ["kept"])

    assert order == ["native", "legacy"]


async def test_post_mcp_call_passthrough_returns_the_result_untouched_without_guardrails(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    response: Final = CallToolResult(content=[TextContent(type="text", text="tool output")])

    result: Final = await proxy_logging.post_mcp_call_hook(response=response, request_data={}, user_api_key_dict=AUTH)

    assert result is response


async def test_post_mcp_call_native_runs_before_legacy(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_McpGuardrail(order)])
    response: Final = CallToolResult(content=[TextContent(type="text", text="tool output")])

    await proxy_logging.post_mcp_call_hook(response=response, request_data={"metadata": {}}, user_api_key_dict=AUTH)

    assert order == ["native", "legacy"]


async def test_deferred_stream_guardrails_passthrough_dispatches_the_guardrail_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: Final[list[str]] = []
    replacement: Final = ModelResponse()
    monkeypatch.setattr(litellm, "callbacks", [_StreamGuardrail(order, replacement)])
    dispatch: Final = _SuccessDispatch()

    await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
        captured_data={"model": "gpt-4o", "metadata": {}},
        captured_user_api_key_dict=AUTH,
        captured_logging_obj=dispatch,
        assembled_response=ModelResponse(),
        cache_hit=None,
    )
    await _drain()

    assert order == ["legacy"]
    assert dispatch.dispatched == [replacement]
    assert dispatch.dispatched[0] is replacement


async def test_deferred_stream_guardrails_native_runs_before_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    order: Final[list[str]] = []
    register_native(build_manifest(_Witness(order)))
    monkeypatch.setattr(litellm, "callbacks", [_StreamGuardrail(order, None)])

    await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
        captured_data={"model": "gpt-4o", "metadata": {}},
        captured_user_api_key_dict=AUTH,
        captured_logging_obj=_SuccessDispatch(),
        assembled_response=ModelResponse(),
        cache_hit=None,
    )

    assert order == ["native", "legacy"]


async def _drain() -> None:
    for _ in range(3):
        await asyncio.sleep(0)
