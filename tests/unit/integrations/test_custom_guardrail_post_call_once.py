import asyncio
import json
from collections.abc import AsyncIterable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.router import Router
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, Choices, GenericGuardrailAPIInputs, LLMResponseTypes, Message, ModelResponse

pytestmark = pytest.mark.respx(assert_all_called=False)

REPLY: Final = "hi from the model"
REWRITTEN: Final = "rewritten by the proxy"
MASKED: Final = "masked by G"

Route = Literal["acompletion", "aresponses", "anthropic_messages"]

_OPENAI_API_BASE: Final = "https://api.openai.com/v1"
_FAILING_API_BASE: Final = "https://primary.invalid/v1"
_MCP_API_BASE: Final = "https://mcp-rounds.invalid/v1"

_CHAT_REPLY: Final[Mapping[str, object]] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_CHAT_SSE: Final = (
    "".join(
        f"data: {json.dumps(chunk)}\n\n"
        for chunk in (
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": REPLY}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        )
    )
    + "data: [DONE]\n\n"
)
_RESPONSES_REPLY: Final[Mapping[str, object]] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-4o",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": REPLY, "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
_RESPONSES_SSE: Final = "".join(
    f"data: {json.dumps(event)}\n\n"
    for event in (
        {"type": "response.created", "sequence_number": 0, "response": {**_RESPONSES_REPLY, "output": []}},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": REPLY,
        },
        {"type": "response.completed", "sequence_number": 2, "response": _RESPONSES_REPLY},
    )
)
_TOOL_CALL_ROUND: Final[Mapping[str, object]] = {
    **_RESPONSES_REPLY,
    "id": "resp_tool_call",
    "output": [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_echo",
            "name": "echo",
            "arguments": '{"value": "PING"}',
            "status": "completed",
        }
    ],
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class _CountedGuardrail(CustomGuardrail):
    checked: tuple[str, ...] = ()


class ApplyGuardrailG(_CountedGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        self.checked = (*self.checked, " ".join(inputs.get("texts") or ()))
        return inputs


def _reply_text(response: LLMResponseTypes) -> str:
    match response:
        case ModelResponse(choices=[Choices(message=Message(content=str() as text))]):
            return text
        case _:
            return str(response)


class OwnHookG(_CountedGuardrail):
    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: LLMResponseTypes
    ) -> None:
        self.checked = (*self.checked, _reply_text(response))


class InheritedApplyGuardrailG(ApplyGuardrailG):
    pass


class MaskingG(ApplyGuardrailG):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        checked: Final = await super().apply_guardrail(inputs, request_data, input_type, logging_obj)
        return {**checked, "texts": [MASKED for _ in checked.get("texts") or ()]}


class BlockingG(ApplyGuardrailG):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        await super().apply_guardrail(inputs, request_data, input_type, logging_obj)
        raise HTTPException(status_code=400, detail="blocked by G")


class RewritingGuardrail(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        return {**inputs, "texts": [REWRITTEN for _ in inputs.get("texts") or ()]}


class ToolCallBlockingGuardrail(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type == "response" and inputs.get("tool_calls"):
            raise HTTPException(status_code=400, detail="tool calls are not allowed")
        return inputs


def _guardrail_g(
    style: type[_CountedGuardrail] = ApplyGuardrailG, default_on: bool = False, run_in_parallel: bool = False
) -> _CountedGuardrail:
    return style(
        guardrail_name="G",
        event_hook=GuardrailEventHooks.post_call,
        default_on=default_on,
        run_in_parallel=run_in_parallel,
    )


def _known_gap(action_item: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(
        raises=AssertionError,
        strict=True,
        reason=f"broken before #40154, out of scope for LIT-10481, tracked as RCA {action_item}",
    )


def _deployment(model_name: str, guardrails: tuple[str, ...], api_base: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "openai/gpt-4o",
            "api_key": "sk-test",
            "api_base": api_base,
            **({"guardrails": list(guardrails)} if guardrails else {}),
        },
    }


def _router(primary: tuple[str, ...], fallback: tuple[str, ...] | None = None, cached: bool = False) -> Router:
    if fallback is None:
        return Router(model_list=[_deployment("gpt", primary, _OPENAI_API_BASE)], cache_responses=cached)
    return Router(
        model_list=[
            _deployment("gpt", primary, _FAILING_API_BASE),
            _deployment("gpt-backup", fallback, _OPENAI_API_BASE),
        ],
        fallbacks=[{"gpt": ["gpt-backup"]}],
        num_retries=0,
    )


def _json_or_sse(json_reply: Mapping[str, object], sse_reply: str) -> Callable[[httpx.Request], httpx.Response]:
    def reply(request: httpx.Request) -> httpx.Response:
        if _JSON_OBJECT.validate_json(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_reply.encode())
        return httpx.Response(200, json=dict(json_reply))

    return reply


async def _drain(stream: AsyncIterable[object]) -> str:
    return "".join([str(chunk) async for chunk in stream])


@pytest.fixture(autouse=True)
def provider(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> Iterator[respx.MockRouter]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    ProxyLogging._callback_capabilities_cache.clear()  # pyright: ignore[reportPrivateUsage]  # keyed on id(), which a collected guardrail from an earlier test can reuse
    for endpoint in ("chat/completions", "responses"):
        respx_mock.post(f"{_FAILING_API_BASE}/{endpoint}").mock(
            return_value=httpx.Response(500, json={"error": {"message": "primary down"}})
        )
    respx_mock.post(f"{_OPENAI_API_BASE}/chat/completions").mock(side_effect=_json_or_sse(_CHAT_REPLY, _CHAT_SSE))
    respx_mock.post(f"{_OPENAI_API_BASE}/responses").mock(side_effect=_json_or_sse(_RESPONSES_REPLY, _RESPONSES_SSE))
    yield respx_mock
    litellm.in_memory_llm_clients_cache.flush_cache()


@dataclass(frozen=True, slots=True)
class ProxyCase:
    route: Route = "acompletion"
    stream: bool = False
    primary: tuple[str, ...] = ("G",)
    fallback: tuple[str, ...] | None = None
    request: tuple[str, ...] = ()
    default_on: bool = False
    style: type[_CountedGuardrail] = ApplyGuardrailG
    cached: bool = False
    client_sees: str = REPLY
    checks: tuple[str, ...] = (REPLY,)


PROXY_CASES: Final = (
    pytest.param(ProxyCase(), id="A-chat"),
    pytest.param(ProxyCase(route="aresponses"), id="A-responses"),
    pytest.param(ProxyCase(route="anthropic_messages"), id="A-messages"),
    pytest.param(ProxyCase(request=("G",)), id="A-chat-model-and-request"),
    pytest.param(ProxyCase(style=MaskingG, client_sees=MASKED), id="A-chat-g-masks-its-own-output"),
    pytest.param(ProxyCase(stream=True), id="B-chat-stream"),
    pytest.param(ProxyCase(route="aresponses", stream=True), id="B-responses-stream"),
    pytest.param(ProxyCase(route="anthropic_messages", stream=True), id="B-messages-stream"),
    pytest.param(ProxyCase(primary=(), fallback=("G",)), id="E-only-on-fallback"),
    pytest.param(
        ProxyCase(request=("rewriter",), client_sees=REWRITTEN, checks=(REPLY, REWRITTEN)),
        id="G-rewritten-after-check",
    ),
    pytest.param(ProxyCase(fallback=("G",)), id="H-on-both-deployments"),
    pytest.param(ProxyCase(primary=(), request=("G",)), id="I-request-chat"),
    pytest.param(ProxyCase(route="aresponses", primary=(), request=("G",)), id="I-request-responses"),
    pytest.param(ProxyCase(style=OwnHookG), id="J-A-chat-own-hook", marks=_known_gap("AI-5")),
    pytest.param(ProxyCase(primary=(), fallback=("G",), style=OwnHookG), id="E-only-on-fallback-own-hook"),
    pytest.param(ProxyCase(primary=(), request=("G",), style=OwnHookG), id="I-request-chat-own-hook"),
    pytest.param(
        ProxyCase(primary=(), request=("G",), style=InheritedApplyGuardrailG),
        id="K-I-request-chat-inherited-apply-guardrail",
    ),
    pytest.param(ProxyCase(primary=("other",), default_on=True), id="L-default-on"),
)

_ROUTE_PATHS: Final[Mapping[Route, str]] = MappingProxyType(
    {
        "acompletion": "/v1/chat/completions",
        "aresponses": "/v1/responses",
        "anthropic_messages": "/v1/messages",
    }
)
_ROUTE_BODIES: Final[Mapping[Route, Mapping[str, object]]] = MappingProxyType(
    {
        "acompletion": {"messages": [{"role": "user", "content": "hello"}]},
        "aresponses": {"input": "hello"},
        "anthropic_messages": {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32},
    }
)


async def _proxy_reply(case: ProxyCase) -> str:
    from litellm.proxy.proxy_server import ProxyConfig, select_data_generator

    path: Final = _ROUTE_PATHS[case.route]
    router: Final = _router(case.primary, case.fallback, case.cached)
    response: Final = await ProxyBaseLLMRequestProcessing(
        data={
            "model": "gpt",
            **_ROUTE_BODIES[case.route],
            **({"stream": True} if case.stream else {}),
            **({"guardrails": list(case.request)} if case.request else {}),
        }
    ).base_process_llm_request(
        request=Request({"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""}),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", request_route=path),
        route_type=case.route,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=UserApiKeyCache()),
        general_settings={},
        proxy_config=ProxyConfig(),
        select_data_generator=select_data_generator,
        llm_router=router,
        is_streaming_request=case.stream,
    )
    text: Final = (
        await _drain(response.body_iterator)
        if isinstance(response, StreamingResponse)
        else json.dumps(response.model_dump() if isinstance(response, BaseModel) else response)
    )
    await asyncio.sleep(0)
    return text


@pytest.mark.parametrize("case", PROXY_CASES)
async def test_a_proxy_request_checks_each_distinct_response_once(
    monkeypatch: pytest.MonkeyPatch, case: ProxyCase
) -> None:
    guardrail: Final = _guardrail_g(case.style, default_on=case.default_on)
    rewriter: Final = RewritingGuardrail(guardrail_name="rewriter", event_hook=GuardrailEventHooks.post_call)
    monkeypatch.setattr(litellm, "callbacks", [rewriter, guardrail])

    reply: Final = await _proxy_reply(case)

    assert case.client_sees in reply
    assert sorted(guardrail.checked) == sorted(case.checks)


@pytest.mark.parametrize("route", ["acompletion", "aresponses"])
async def test_a_cache_hit_is_checked_like_a_fresh_response(
    monkeypatch: pytest.MonkeyPatch, provider: respx.MockRouter, route: Route
) -> None:
    guardrail: Final = _guardrail_g()
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))

    case: Final = ProxyCase(route=route, cached=True)
    replies: Final = (await _proxy_reply(case), await _proxy_reply(case))

    assert provider.calls.call_count == 1, "the second request must be served from the cache"
    assert all(REPLY in reply for reply in replies)
    assert guardrail.checked == (REPLY, REPLY)


async def test_a_blocking_verdict_blocks_the_proxy_response_after_one_check(monkeypatch: pytest.MonkeyPatch) -> None:
    guardrail: Final = _guardrail_g(BlockingG)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    with pytest.raises(HTTPException, check=lambda e: e.detail == "blocked by G"):
        await _proxy_reply(ProxyCase())

    assert guardrail.checked == (REPLY,)


SdkCaller = Literal["sdk-call", "sdk-router"]


@dataclass(frozen=True, slots=True)
class SdkCase:
    caller: SdkCaller = "sdk-call"
    stream: bool = False
    style: type[_CountedGuardrail] = ApplyGuardrailG


SDK_CASES: Final = (
    pytest.param(SdkCase(), id="C-sdk-call"),
    pytest.param(SdkCase(caller="sdk-router"), id="C-sdk-router"),
    pytest.param(SdkCase(style=OwnHookG), id="C-sdk-call-own-hook"),
    pytest.param(SdkCase(stream=True), id="D-sdk-call-stream", marks=_known_gap("AI-4")),
)


async def _sdk_reply(case: SdkCase) -> str:
    messages: Final = [{"role": "user", "content": "hello"}]
    response: Final = await (
        _router(("G",)).acompletion(model="gpt", messages=messages, stream=case.stream)
        if case.caller == "sdk-router"
        else litellm.acompletion(
            model="openai/gpt-4o", api_key="sk-test", guardrails=["G"], messages=messages, stream=case.stream
        )
    )
    text: Final = await _drain(response) if isinstance(response, AsyncIterable) else str(response)
    await asyncio.sleep(0)
    return text


@pytest.mark.parametrize("case", SDK_CASES)
async def test_an_sdk_call_checks_its_response_once(monkeypatch: pytest.MonkeyPatch, case: SdkCase) -> None:
    guardrail: Final = _guardrail_g(case.style)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    reply: Final = await _sdk_reply(case)

    assert REPLY in reply
    assert guardrail.checked == (REPLY,)


async def test_mcp_auto_execute_checks_the_tool_call_round_before_running_its_tools(
    monkeypatch: pytest.MonkeyPatch, provider: respx.MockRouter
) -> None:
    monkeypatch.setattr(
        litellm, "callbacks", [ToolCallBlockingGuardrail(guardrail_name="G", event_hook=GuardrailEventHooks.post_call)]
    )
    provider.post(f"{_MCP_API_BASE}/responses").mock(
        side_effect=[httpx.Response(200, json=_TOOL_CALL_ROUND), httpx.Response(200, json=_RESPONSES_REPLY)]
    )
    echo_tool: Final = MCPTool.model_validate(
        {"name": "echo", "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}}},
        by_name=False,
    )
    run_tools: Final = AsyncMock(return_value=[{"tool_call_id": "call_echo", "result": "PING", "name": "echo"}])

    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            AsyncMock(return_value=([echo_tool], {"echo": "demo"})),
        ),
        patch.object(LiteLLM_Proxy_MCP_Handler, "execute_tool_calls", run_tools),
        pytest.raises(HTTPException, check=lambda e: e.detail == "tool calls are not allowed"),
    ):
        await litellm.aresponses(
            model="openai/gpt-4o",
            input="Call echo with PING",
            tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp/demo", "require_approval": "never"}],
            api_key="sk-test",
            api_base=_MCP_API_BASE,
            guardrails=["G"],
            proxy_server_request={"url": "http://localhost:4000/v1/responses", "method": "POST"},
            litellm_metadata={"user_api_key_request_route": "/v1/responses"},
        )

    run_tools.assert_not_awaited()


def _same_response(reply: ModelResponse) -> ModelResponse:
    return reply


def _model_renamed(reply: ModelResponse) -> ModelResponse:
    return reply.model_copy(update={"model": "public-alias"})


def _rewritten(_: ModelResponse) -> ModelResponse:
    return ModelResponse(choices=[{"message": {"role": "assistant", "content": REWRITTEN}}])


PROXY_LOOP_CASES: Final = (
    pytest.param(_same_response, (REPLY,), id="A-same-response"),
    pytest.param(_model_renamed, (REPLY,), id="A-model-renamed"),
    pytest.param(_rewritten, (REPLY, REWRITTEN), id="G-rewritten"),
)


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
@pytest.mark.parametrize(("proxy_sees", "checks"), PROXY_LOOP_CASES)
async def test_each_proxy_post_call_loop_skips_only_output_the_deployment_checked(
    monkeypatch: pytest.MonkeyPatch,
    parallel: bool,
    proxy_sees: Callable[[ModelResponse], ModelResponse],
    checks: tuple[str, ...],
) -> None:
    guardrail: Final = _guardrail_g(run_in_parallel=parallel)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    data: Final = {
        "model": "gpt",
        "proxy_server_request": {"url": "http://localhost:4000/v1/chat/completions"},
        "metadata": {"guardrails": ["G"]},
    }
    reply: Final = ModelResponse(model="provider-model", choices=[{"message": {"role": "assistant", "content": REPLY}}])

    await guardrail.async_post_call_success_deployment_hook(
        request_data={**data, "guardrails": ["G"]}, response=reply, call_type=CallTypes.acompletion
    )
    await ProxyLogging(user_api_key_cache=UserApiKeyCache()).post_call_success_hook(
        data=data, response=proxy_sees(reply), user_api_key_dict=UserAPIKeyAuth()
    )

    assert sorted(guardrail.checked) == sorted(checks)


def _ocr_reply(markdown: str) -> OCRResponse:
    return OCRResponse.model_validate(
        {"pages": [{"index": 0, "markdown": markdown, "images": [], "dimensions": None}], "model": "mistral-ocr-latest"}
    )


@pytest.mark.parametrize(
    ("proxy_sees", "checks"),
    [
        pytest.param(REPLY, (REPLY,), id="A-ocr-same-pages"),
        pytest.param(REWRITTEN, (REPLY, REWRITTEN), id="G-ocr-rewritten-pages"),
    ],
)
async def test_an_ocr_response_is_skipped_only_when_its_pages_were_checked(
    monkeypatch: pytest.MonkeyPatch, proxy_sees: str, checks: tuple[str, ...]
) -> None:
    guardrail: Final = _guardrail_g()
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    data: Final = {
        "model": "ocr",
        "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"},
        "proxy_server_request": {"url": "http://localhost:4000/v1/ocr"},
        "metadata": {"guardrails": ["G"], "user_api_key_request_route": "/v1/ocr"},
    }

    await guardrail.async_post_call_success_deployment_hook(
        request_data={**data, "guardrails": ["G"]}, response=_ocr_reply(REPLY), call_type=CallTypes.aocr
    )
    await ProxyLogging(user_api_key_cache=UserApiKeyCache()).post_call_success_hook(
        data=data, response=_ocr_reply(proxy_sees), user_api_key_dict=UserAPIKeyAuth(request_route="/v1/ocr")
    )

    assert sorted(guardrail.checked) == sorted(checks)
