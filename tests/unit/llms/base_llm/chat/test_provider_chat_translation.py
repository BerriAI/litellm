import base64
import copy
import itertools
import json
import struct
import zlib
from typing import Callable, Final, Iterable, Literal, Mapping, cast

import httpx
import pytest
import respx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from respx import MockRouter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
)
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.types.utils import CallTypes
from litellm.utils import ProviderConfigManager, return_raw_request

_Shape = Literal[
    "openai",
    "anthropic",
    "gemini",
    "bedrock_converse",
    "bedrock_invoke",
    "bedrock_invoke_nova",
    "bedrock_invoke_openai",
]

_AWS_KWARGS: Final[Mapping[str, str]] = {
    "aws_access_key_id": "AKIAFAKE",
    "aws_secret_access_key": "fakesecret",
    "aws_region_name": "us-east-1",
}


class _Kwargs(TypedDict, total=False):
    model: ReadOnly[str]
    api_key: ReadOnly[str]
    api_base: ReadOnly[str]
    api_version: ReadOnly[str]
    aws_access_key_id: ReadOnly[str]
    aws_secret_access_key: ReadOnly[str]
    aws_region_name: ReadOnly[str]


class _Case(TypedDict):
    id: ReadOnly[str]
    shape: ReadOnly[_Shape]
    kwargs: ReadOnly[_Kwargs]
    url: ReadOnly[str]
    stream_url: ReadOnly[str]
    router: ReadOnly[bool]


def _converse_url(model_id: str, region: str = "us-east-1") -> str:
    return f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"


def _converse_stream_url(model_id: str, region: str = "us-east-1") -> str:
    return f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse-stream"


def _invoke_url(model_id: str) -> str:
    return f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/invoke"


def _invoke_stream_url(model_id: str) -> str:
    return f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/invoke-with-response-stream"


def _bedrock_case(
    case_id: str,
    model: str,
    invoke_model_id: str,
    shape: _Shape,
    region: str = "us-east-1",
) -> _Case:
    kwargs: _Kwargs = {"model": model, **cast(_Kwargs, dict(_AWS_KWARGS))}
    if region != "us-east-1":
        kwargs = {"model": model, **cast(_Kwargs, dict(_AWS_KWARGS)), "aws_region_name": region}
    is_invoke = "invoke" in model
    url: Final = _invoke_url(invoke_model_id) if is_invoke else _converse_url(invoke_model_id, region)
    stream_url: Final = (
        _invoke_stream_url(invoke_model_id) if is_invoke else _converse_stream_url(invoke_model_id, region)
    )
    return {
        "id": case_id,
        "shape": shape,
        "kwargs": kwargs,
        "url": url,
        "stream_url": stream_url,
        "router": False,
    }


def _openai_case(
    case_id: str,
    model: str,
    url: str,
    *,
    router: bool = False,
    extra: _Kwargs | None = None,
) -> _Case:
    kwargs: _Kwargs = {"model": model, "api_key": "sk-offline"}
    if extra is not None:
        kwargs = {**kwargs, **extra}
    return {
        "id": case_id,
        "shape": "openai",
        "kwargs": kwargs,
        "url": url,
        "stream_url": url,
        "router": router,
    }


_CASES: Final[tuple[_Case, ...]] = (
    _openai_case("openai_gpt4omini", "gpt-4o-mini", "https://api.openai.com/v1/chat/completions"),
    _openai_case("router_gpt4omini", "gpt-4o-mini", "https://api.openai.com/v1/chat/completions", router=True),
    _openai_case("openai_o1", "o1", "https://api.openai.com/v1/chat/completions"),
    _openai_case("openai_o3mini", "o3-mini", "https://api.openai.com/v1/chat/completions"),
    _openai_case(
        "azure_o3mini",
        "azure/o3-mini",
        "https://openai-gpt-4-test-v-1.openai.azure.com/openai/deployments/o3-mini/chat/completions?api-version=2024-02-15-preview",
        extra={
            "api_key": "k",
            "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
            "api_version": "2024-02-15-preview",
        },
    ),
    _openai_case(
        "azure_o3mini_live",
        "azure/o3-mini",
        "https://openai-prod-test.openai.azure.com/openai/deployments/o3-mini/chat/completions?api-version=2024-12-01-preview",
        extra={
            "api_key": "k",
            "api_base": "https://openai-prod-test.openai.azure.com",
            "api_version": "2024-12-01-preview",
        },
    ),
    {
        "id": "anthropic_sonnet45",
        "shape": "anthropic",
        "kwargs": {"model": "anthropic/claude-sonnet-4-5-20250929", "api_key": "sk-offline"},
        "url": "https://api.anthropic.com/v1/messages",
        "stream_url": "https://api.anthropic.com/v1/messages",
        "router": False,
    },
    {
        "id": "gemini_25flash",
        "shape": "gemini",
        "kwargs": {"model": "gemini/gemini-2.5-flash", "api_key": "k"},
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "stream_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        "router": False,
    },
    _openai_case("mistral_medium", "mistral/mistral-medium-latest", "https://api.mistral.ai/v1/chat/completions"),
    _openai_case("together_glm", "together_ai/zai-org/GLM-5.3-Flash", "https://api.together.ai/v1/chat/completions"),
    _openai_case("groq_oss120b", "groq/openai/gpt-oss-120b", "https://api.groq.com/openai/v1/chat/completions"),
    _openai_case("xai_grok3mini", "xai/grok-3-mini-beta", "https://api.x.ai/v1/chat/completions"),
    _openai_case(
        "huggingface_llama",
        "huggingface/together/meta-llama/Meta-Llama-3-8B-Instruct",
        "https://router.huggingface.co/together/v1/chat/completions",
        extra={"api_base": "https://router.huggingface.co/together/v1"},
    ),
    _bedrock_case(
        "bedrock_converse_haiku",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock_converse",
    ),
    _bedrock_case(
        "bedrock_converse_haiku_xregion",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock_converse",
        region="us-west-2",
    ),
    _bedrock_case(
        "bedrock_converse_novalite", "bedrock/us.amazon.nova-lite-v1:0", "us.amazon.nova-lite-v1:0", "bedrock_converse"
    ),
    _bedrock_case(
        "bedrock_converse_novamicro",
        "bedrock/converse/us.amazon.nova-micro-v1:0",
        "us.amazon.nova-micro-v1:0",
        "bedrock_converse",
    ),
    _bedrock_case(
        "bedrock_converse_llama33",
        "bedrock/converse/us.meta.llama3-3-70b-instruct-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
        "bedrock_converse",
    ),
    _bedrock_case(
        "bedrock_converse_gptoss",
        "bedrock/converse/openai.gpt-oss-20b-1:0",
        "openai.gpt-oss-20b-1:0",
        "bedrock_converse",
    ),
    _bedrock_case(
        "bedrock_converse_anthropic_thinking",
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock_converse",
    ),
    _bedrock_case(
        "bedrock_invoke_haiku",
        "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock_invoke",
    ),
    _bedrock_case(
        "bedrock_invoke_novamicro",
        "bedrock/invoke/us.amazon.nova-micro-v1:0",
        "us.amazon.nova-micro-v1:0",
        "bedrock_invoke_nova",
    ),
    _bedrock_case(
        "bedrock_invoke_kimi",
        "bedrock/invoke/moonshot.kimi-k2-thinking",
        "moonshot.kimi-k2-thinking",
        "bedrock_invoke_openai",
    ),
)

_BY_ID: Final[Mapping[str, _Case]] = {c["id"]: c for c in _CASES}


def _case_id(case: _Case) -> str:
    return case["id"]


def _pick(*ids: str) -> tuple[_Case, ...]:
    return tuple(_BY_ID[i] for i in ids)


_JSON: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(list[JsonValue])


def _openai_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "service_tier": "default",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


def _anthropic_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_offline",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


def _gemini_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": text}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        },
    )


def _converse_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "metrics": {"latencyMs": 1},
        },
    )


def _canned_response(case: _Case, text: str) -> httpx.Response:
    match case["shape"]:
        case "openai" | "bedrock_invoke_openai":
            return _openai_response(text)
        case "anthropic" | "bedrock_invoke":
            return _anthropic_response(text)
        case "gemini":
            return _gemini_response(text)
        case "bedrock_converse" | "bedrock_invoke_nova":
            return _converse_response(text)


_PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_PNG_B64: Final = base64.b64encode(_PNG_BYTES).decode()
_PNG_URLS: Final = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm@d769e81c90d453240c61fc572cdb27fae06a89d0/ui/litellm-dashboard/public/assets/logos/litellm_logo.jpg",
    "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png",
)


def _aws_str_header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return (
        struct.pack("!B", len(name_bytes))
        + name_bytes
        + struct.pack("!B", 7)
        + struct.pack("!H", len(value_bytes))
        + value_bytes
    )


def _aws_frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    payload_bytes: Final = json.dumps(payload).encode()
    headers_bytes: Final = b"".join(
        (
            _aws_str_header(":event-type", event_type),
            _aws_str_header(":content-type", "application/json"),
            _aws_str_header(":message-type", "event"),
        )
    )
    prelude: Final = struct.pack("!II", 12 + len(headers_bytes) + len(payload_bytes) + 4, len(headers_bytes))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF) + headers_bytes + payload_bytes
    return message + struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)


def _openai_sse(text: str) -> str:
    chunks: Final = (
        {
            "id": "chatcmpl-offline",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            "service_tier": "default",
        },
        {
            "id": "chatcmpl-offline",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "service_tier": "default",
        },
    )
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _anthropic_sse(text: str) -> str:
    events: Final = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_offline",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "m",
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events)


def _gemini_sse(text: str) -> str:
    event: Final = {
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }
    return f"data: {json.dumps(event)}\n\n"


def _converse_stream_bytes(text: str) -> bytes:
    frames: Final = (
        _aws_frame("messageStart", {"role": "assistant"}),
        _aws_frame("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 0}),
        _aws_frame("contentBlockStop", {"contentBlockIndex": 0}),
        _aws_frame("messageStop", {"stopReason": "end_turn"}),
        _aws_frame("metadata", {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}),
    )
    return b"".join(frames)


def _invoke_stream_bytes(text: str) -> bytes:
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_invoke",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "m",
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )
    return b"".join(_aws_frame("chunk", {"bytes": base64.b64encode(json.dumps(e).encode()).decode()}) for e in events)


def _invoke_nova_stream_bytes(text: str) -> bytes:
    events: Final = (
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}},
    )
    return b"".join(_aws_frame("chunk", {"bytes": base64.b64encode(json.dumps(e).encode()).decode()}) for e in events)


def _stream_canned(case: _Case, text: str) -> httpx.Response:
    match case["shape"]:
        case "openai" | "bedrock_invoke_openai":
            return httpx.Response(200, content=_openai_sse(text), headers={"content-type": "text/event-stream"})
        case "anthropic":
            return httpx.Response(200, content=_anthropic_sse(text), headers={"content-type": "text/event-stream"})
        case "gemini":
            return httpx.Response(200, content=_gemini_sse(text), headers={"content-type": "text/event-stream"})
        case "bedrock_converse":
            return httpx.Response(
                200,
                content=_converse_stream_bytes(text),
                headers={"content-type": "application/vnd.amazon.eventstream"},
            )
        case "bedrock_invoke":
            return httpx.Response(
                200,
                content=_invoke_stream_bytes(text),
                headers={"content-type": "application/vnd.amazon.eventstream"},
            )
        case "bedrock_invoke_nova":
            return httpx.Response(
                200,
                content=_invoke_nova_stream_bytes(text),
                headers={"content-type": "application/vnd.amazon.eventstream"},
            )


@pytest.fixture(autouse=True)
def _httpx_only_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


def _tool_use_response(case: _Case, payload: str) -> httpx.Response:
    arguments: Final = json.loads(payload)
    match case["shape"]:
        case "anthropic" | "bedrock_invoke":
            return httpx.Response(
                200,
                json={
                    "id": "msg_offline",
                    "type": "message",
                    "role": "assistant",
                    "model": "m",
                    "content": [
                        {"type": "tool_use", "id": "toolu_offline", "name": "json_tool_call", "input": arguments}
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        case "bedrock_converse" | "bedrock_invoke_nova":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "toolUse": {
                                        "toolUseId": "tooluse_offline",
                                        "name": "json_tool_call",
                                        "input": arguments,
                                    }
                                }
                            ],
                        }
                    },
                    "stopReason": "tool_use",
                    "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                    "metrics": {"latencyMs": 1},
                },
            )
        case _:
            return _canned_response(case, payload)


def _json_responder(case: _Case, payload: str) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        if b"json_tool_call" in request.content:
            return _tool_use_response(case, payload)
        return _canned_response(case, payload)

    return respond


def _register(
    case: _Case,
    respx_mock: MockRouter,
    text: str,
    *,
    stream: bool = False,
    json_via_tool: bool = False,
) -> respx.Route:
    for url in _PNG_URLS:
        respx_mock.get(url).mock(return_value=httpx.Response(200, content=_PNG_BYTES))
    target: Final = case["stream_url"] if stream else case["url"]
    if json_via_tool:
        return respx_mock.post(target).mock(side_effect=_json_responder(case, text))
    if case["shape"] == "gemini" and stream:
        return respx_mock.post(url__startswith=target).mock(return_value=_stream_canned(case, text))
    if stream and case["id"].startswith("groq"):
        return respx_mock.post(target).mock(return_value=_canned_response(case, text))
    return respx_mock.post(target).mock(
        return_value=_stream_canned(case, text) if stream else _canned_response(case, text)
    )


_ROUTER_MODEL_LIST: Final = [
    {
        "model_name": "offline-router-model",
        "litellm_params": {"model": "gpt-4o-mini", "api_key": "sk-offline"},
    }
]


def _call(case: _Case, params: Mapping[str, JsonValue]) -> litellm.ModelResponse | litellm.CustomStreamWrapper:
    extra: Final = copy.deepcopy(params)
    if case["router"]:
        router: Final = litellm.Router(model_list=copy.deepcopy(_ROUTER_MODEL_LIST))
        return router.completion(model="offline-router-model", **extra)
    return litellm.completion(**case["kwargs"], **extra)


def _complete(case: _Case, params: Mapping[str, JsonValue]) -> litellm.ModelResponse:
    response: Final = _call(case, params)
    assert isinstance(response, litellm.ModelResponse)
    return response


def _stream(case: _Case, params: Mapping[str, JsonValue]) -> litellm.CustomStreamWrapper:
    response: Final = _call(case, {**params, "stream": True})
    assert isinstance(response, litellm.CustomStreamWrapper)
    return response


async def _acomplete(case: _Case, params: Mapping[str, JsonValue]) -> litellm.ModelResponse:
    extra: Final = copy.deepcopy(params)
    if case["router"]:
        router: Final = litellm.Router(model_list=copy.deepcopy(_ROUTER_MODEL_LIST))
        response: Final = await router.acompletion(model="offline-router-model", **extra)
    else:
        response = await litellm.acompletion(**case["kwargs"], **extra)
    assert isinstance(response, litellm.ModelResponse)
    return response


def _request_body(route: respx.Route) -> Mapping[str, JsonValue]:
    assert route.calls, "provider route was never called"
    return _JSON.validate_python(json.loads(route.calls.last.request.content))


def _mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    return _JSON.validate_python(value)


def _items(value: JsonValue) -> tuple[JsonValue, ...]:
    return tuple(_ITEMS.validate_python(value))


def _mappings(value: JsonValue) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(_mapping(item) for item in _items(value))


def _flatten(groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    return tuple(itertools.chain.from_iterable(groups))


def _typed_text_parts(content: JsonValue) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,)
    return tuple(str(p["text"]) for p in _mappings(content) if p.get("type") == "text")


def _keyed_text_parts(content: JsonValue) -> tuple[str, ...]:
    return tuple(str(p["text"]) for p in _mappings(content) if "text" in p)


def _user_texts(case: _Case, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    match case["shape"]:
        case "openai" | "bedrock_invoke_openai":
            return _flatten(_typed_text_parts(m["content"]) for m in _mappings(body["messages"]) if m["role"] == "user")
        case "anthropic" | "bedrock_invoke":
            return _flatten(_typed_text_parts(m["content"]) for m in _mappings(body["messages"]) if m["role"] == "user")
        case "bedrock_converse" | "bedrock_invoke_nova":
            return _flatten(_keyed_text_parts(m["content"]) for m in _mappings(body["messages"]) if m["role"] == "user")
        case "gemini":
            return _flatten(
                _keyed_text_parts(m["parts"]) for m in _mappings(body["contents"]) if m.get("role") != "model"
            )


def _system_texts(case: _Case, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    match case["shape"]:
        case "openai" | "bedrock_invoke_openai":
            return tuple(str(m["content"]) for m in _mappings(body["messages"]) if m["role"] == "system")
        case "anthropic" | "bedrock_invoke" | "bedrock_converse" | "bedrock_invoke_nova":
            return tuple(str(s["text"]) for s in _mappings(body.get("system", [])))
        case "gemini":
            return tuple(
                str(p["text"]) for p in _mappings(_mapping(body.get("system_instruction", {})).get("parts", []))
            )


def _message_roles(case: _Case, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    match case["shape"]:
        case "gemini":
            return tuple(str(m.get("role", "user")) for m in _mappings(body["contents"]))
        case _:
            return tuple(str(m["role"]) for m in _mappings(body["messages"]))


def _tools_payload(case: _Case, body: Mapping[str, JsonValue]) -> JsonValue:
    match case["shape"]:
        case "openai" | "bedrock_invoke_openai" | "anthropic" | "bedrock_invoke":
            return body.get("tools")
        case "bedrock_converse" | "bedrock_invoke_nova":
            return _mapping(body.get("toolConfig", {})).get("tools")
        case "gemini":
            declared: Final = _mappings(body.get("tools", []))
            if not declared:
                return None
            return declared[0].get("function_declarations")


def _tool_names(case: _Case, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    entries: Final = _mappings(_tools_payload(case, body) or [])
    match case["shape"]:
        case "openai" | "bedrock_invoke_openai":
            return tuple(str(_mapping(e["function"])["name"]) for e in entries)
        case "bedrock_converse" | "bedrock_invoke_nova":
            return tuple(str(_mapping(e["toolSpec"])["name"]) for e in entries)
        case "anthropic" | "bedrock_invoke" | "gemini":
            return tuple(str(e["name"]) for e in entries)


def _tool_input_schema(case: _Case, body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    first: Final = _mappings(_tools_payload(case, body) or [])[0]
    match case["shape"]:
        case "openai" | "bedrock_invoke_openai":
            return _mapping(_mapping(first["function"])["parameters"])
        case "anthropic" | "bedrock_invoke":
            return _mapping(first["input_schema"])
        case "bedrock_converse" | "bedrock_invoke_nova":
            return _mapping(_mapping(_mapping(first["toolSpec"])["inputSchema"])["json"])
        case "gemini":
            return _mapping(first["parameters"])


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "bedrock_converse_haiku",
        "bedrock_converse_novalite",
        "bedrock_converse_gptoss",
        "bedrock_converse_llama33",
        "bedrock_invoke_haiku",
        "bedrock_invoke_novamicro",
        "groq_oss120b",
        "huggingface_llama",
        "mistral_medium",
        "openai_gpt4omini",
        "router_gpt4omini",
        "together_glm",
        "xai_grok3mini",
    ),
    ids=_case_id,
)
def test_developer_role_translation(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(
        case,
        {
            "messages": [
                {"role": "developer", "content": "Be a good bot!"},
                {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]},
            ]
        },
    )
    body: Final = _request_body(route)
    assert "developer" not in _message_roles(case, body)
    assert _system_texts(case, body) == ("Be a good bot!",)
    assert "Hello, how are you?" in _user_texts(case, body)
    assert response.choices[0].message.content == f"canned-{case['id']}"


@pytest.mark.parametrize(
    "case",
    _pick(
        "bedrock_converse_gptoss",
        "bedrock_invoke_novamicro",
        "huggingface_llama",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
        "together_glm",
        "xai_grok3mini",
    ),
    ids=_case_id,
)
def test_content_list_handling(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(
        case,
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}]},
    )
    body: Final = _request_body(route)
    assert _user_texts(case, body) == ("Hello, how are you?",)
    first_content: Final = _mappings(body["messages"])[0]["content"]
    if case["id"] == "mistral_medium":
        assert first_content == "Hello, how are you?"
    else:
        assert isinstance(first_content, list)
    assert response.choices[0].message.content == f"canned-{case['id']}"


_TOOL_ARRAY_SCHEMA: Final[Mapping[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "shoe_get_id",
        "description": "Get information about a show by its ID or name",
        "parameters": {
            "type": "object",
            "properties": {"shoe_id": {"type": ["string", "number"], "description": "The shoe ID or name"}},
            "required": ["shoe_id"],
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
        },
    },
}


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_gptoss",
        "bedrock_converse_llama33",
        "bedrock_invoke_haiku",
        "bedrock_invoke_kimi",
        "gemini_25flash",
        "groq_oss120b",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o3mini",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_tool_call_with_property_type_array(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(
        case,
        {
            "messages": [{"role": "user", "content": "Tell me about shoes"}],
            "tools": [_TOOL_ARRAY_SCHEMA],
        },
    )
    body: Final = _request_body(route)
    assert _tool_names(case, body) == ("shoe_get_id",)
    schema: Final = _tool_input_schema(case, body)
    shoe_id: Final = _mapping(_mapping(schema["properties"])["shoe_id"])
    assert schema["required"] == ["shoe_id"]
    if case["shape"] == "gemini":
        assert [_mapping(v)["type"] for v in _items(shoe_id["anyOf"])] == ["string", "number"]
    else:
        assert shoe_id["type"] == ["string", "number"]
    assert response.choices[0].message.content == f"canned-{case['id']}"


_TOOL_ENUM_SCHEMA: Final[Mapping[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "litellm_product_search",
        "description": "Search for product information",
        "parameters": {
            "properties": {
                "search_mode": {
                    "default": "",
                    "description": "The search strategy to use",
                    "enum": ["", "product_search", "product_search_with_filters"],
                    "type": "string",
                }
            },
            "required": ["search_mode"],
            "type": "object",
        },
    },
}


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_gptoss",
        "bedrock_converse_llama33",
        "bedrock_invoke_haiku",
        "bedrock_invoke_kimi",
        "gemini_25flash",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o3mini",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_tool_call_with_empty_enum_property(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(
        case,
        {
            "messages": [{"role": "user", "content": "Search for the latest iPhone models"}],
            "tools": [_TOOL_ENUM_SCHEMA],
        },
    )
    body: Final = _request_body(route)
    assert _tool_names(case, body) == ("litellm_product_search",)
    schema: Final = _tool_input_schema(case, body)
    search_mode: Final = _mapping(_mapping(schema["properties"])["search_mode"])
    enum_values: Final = _items(search_mode["enum"])
    assert len(enum_values) == 3
    assert enum_values[0] == (None if case["shape"] == "gemini" else "")
    assert enum_values[1:] == ("product_search", "product_search_with_filters")
    assert response.choices[0].message.content == f"canned-{case['id']}"


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_converse_gptoss",
        "bedrock_converse_llama33",
        "bedrock_invoke_haiku",
        "gemini_25flash",
        "groq_oss120b",
        "huggingface_llama",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
        "together_glm",
        "xai_grok3mini",
    ),
    ids=_case_id,
)
def test_pydantic_model_input(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    messages: Final = [litellm.Message(content="Hello, how are you?", role="user")]
    response: Final = _complete(case, {"messages": messages})
    assert "Hello, how are you?" in _user_texts(case, _request_body(route))
    assert response.choices[0].message.content == f"canned-{case['id']}"


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_invoke_haiku",
        "gemini_25flash",
        "openai_gpt4omini",
        "openai_o1",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_file_data_unit_test(case: _Case, respx_mock: MockRouter) -> None:
    pdf_b64: Final = base64.b64encode(b"%PDF-1.4 offline dummy").decode()
    file_data_url: Final = f"data:application/pdf;base64,{pdf_b64}"
    raw_request: Final = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            **{k: v for k, v in dict(case["kwargs"]).items() if k != "api_key"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this file about?"},
                        {"type": "file", "file": {"file_data": file_data_url}},
                    ],
                }
            ],
        },
    )
    assert raw_request.get("error") is None
    assert pdf_b64 in json.dumps(raw_request.get("raw_request_body"))


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_converse_gptoss",
        "bedrock_converse_llama33",
        "bedrock_invoke_haiku",
        "gemini_25flash",
        "groq_oss120b",
        "huggingface_llama",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
        "together_glm",
        "xai_grok3mini",
    ),
    ids=_case_id,
)
def test_message_with_name(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(case, {"messages": [{"role": "user", "content": "Hello", "name": "test_name"}]})
    body: Final = _request_body(route)
    assert "Hello" in _user_texts(case, body)
    if case["shape"] in ("openai", "bedrock_invoke_openai"):
        first: Final = cast(Mapping[str, JsonValue], cast(list[JsonValue], body["messages"])[0])
        if case["id"] == "mistral_medium":
            assert "name" not in first
        else:
            assert first.get("name") == "test_name"
    else:
        assert "test_name" not in json.dumps(body)
    assert response.choices[0].message.content == f"canned-{case['id']}"


@pytest.mark.parametrize("response_format", ({"type": "json_object"}, {"type": "text"}))
@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_invoke_haiku",
        "bedrock_invoke_kimi",
        "gemini_25flash",
        "groq_oss120b",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_json_response_format(case: _Case, response_format: Mapping[str, JsonValue], respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, '{"city":"San Francisco","state":"CA"}')
    response: Final = _complete(
        case,
        {
            "messages": [
                {"role": "system", "content": "Your output should be a JSON object with no additional properties."},
                {"role": "user", "content": "Respond with this in json. city=San Francisco, state=CA"},
            ],
            "response_format": response_format,
        },
    )
    body: Final = _request_body(route)
    match case["shape"]:
        case "gemini":
            mime_types: Final = {"json_object": "application/json", "text": "text/plain"}
            config: Final = _mapping(body["generationConfig"])
            assert config["response_mime_type"] == mime_types[str(response_format["type"])]
        case "openai" | "bedrock_invoke_openai":
            assert body["response_format"] == response_format
        case _:
            assert "response_format" not in body
    assert not _tools_payload(case, body)
    assert response.choices[0].message.content == '{"city":"San Francisco","state":"CA"}'


_WEATHER_TOOL: Final[Mapping[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_converse_gptoss",
        "bedrock_converse_llama33",
        "bedrock_invoke_haiku",
        "gemini_25flash",
        "groq_oss120b",
        "huggingface_llama",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
        "together_glm",
        "xai_grok3mini",
    ),
    ids=_case_id,
)
def test_response_format_type_text_with_tool_calls_no_tool_choice(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(
        case,
        {
            "messages": [{"role": "user", "content": "What's the weather like in Boston today?"}],
            "response_format": {"type": "text"},
            "tools": [_WEATHER_TOOL],
            "drop_params": True,
        },
    )
    body: Final = _request_body(route)
    assert _tool_names(case, body) == ("get_current_weather",)
    assert "tool_choice" not in body
    assert "toolChoice" not in _mapping(body.get("toolConfig", {}))
    assert response.choices[0].message.content == f"canned-{case['id']}"


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=_case_id,
)
def test_response_format_type_text(case: _Case) -> None:
    _, provider, _, _ = litellm.get_llm_provider(model=case["kwargs"]["model"])
    provider_config: Final = ProviderConfigManager.get_provider_chat_config(
        case["kwargs"]["model"], litellm.LlmProviders(provider)
    )
    translated_params: Final = provider_config.map_openai_params(
        non_default_params={"response_format": {"type": "text"}},
        optional_params={},
        model=case["kwargs"]["model"],
        drop_params=False,
    )
    assert "tool_choice" not in translated_params
    assert "tools" not in translated_params


class _FirstResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    first_response: str


class _CalendarEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    date: str
    participants: list[str]


class _EventsList(BaseModel):
    model_config = ConfigDict(frozen=True)
    events: list[_CalendarEvent]


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_converse_gptoss",
        "bedrock_invoke_haiku",
        "bedrock_invoke_novamicro",
        "bedrock_invoke_kimi",
        "groq_oss120b",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_json_response_pydantic_obj(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, '{"first_response":"paris"}', json_via_tool=True)
    response: Final = _complete(
        case,
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "response_format": _FirstResponse,
        },
    )
    body: Final = _request_body(route)
    serialized: Final = json.dumps(body)
    assert "first_response" in serialized
    assert json.loads(response.choices[0].message.content) == {"first_response": "paris"}
    assert response.choices[0].message.tool_calls is None


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_converse_gptoss",
        "bedrock_invoke_haiku",
        "bedrock_invoke_novamicro",
        "bedrock_invoke_kimi",
        "bedrock_invoke_novamicro",
        "groq_oss120b",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_json_response_nested_pydantic_obj(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, '{"events":[]}', json_via_tool=True)
    response: Final = _complete(
        case,
        {
            "messages": [{"role": "user", "content": "List 5 important events in the XIX century"}],
            "response_format": _EventsList,
        },
    )
    body: Final = _request_body(route)
    serialized: Final = json.dumps(body)
    assert "events" in serialized
    assert "participants" in serialized
    assert json.loads(response.choices[0].message.content) == {"events": []}
    assert response.choices[0].message.tool_calls is None


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_converse_gptoss",
        "bedrock_invoke_haiku",
        "bedrock_invoke_novamicro",
        "bedrock_invoke_kimi",
        "bedrock_invoke_novamicro",
        "groq_oss120b",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_json_response_nested_json_schema(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, '{"events":[]}', json_via_tool=True)
    response: Final = _complete(
        case,
        {
            "messages": [{"role": "user", "content": "List 5 important events in the XIX century"}],
            "response_format": type_to_response_format_param(_EventsList),
        },
    )
    body: Final = _request_body(route)
    serialized: Final = json.dumps(body)
    assert "events" in serialized
    assert "participants" in serialized
    assert json.loads(response.choices[0].message.content) == {"events": []}
    assert response.choices[0].message.tool_calls is None


def test_audio_input_gemini(respx_mock: MockRouter) -> None:
    case: Final = _BY_ID["gemini_25flash"]
    wav_b64: Final = base64.b64encode(b"RIFFFAKEWAVDATA").decode()
    route: Final = _register(case, respx_mock, "canned-gemini")
    response: Final = _complete(
        case,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this recording?"},
                        {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
                    ],
                }
            ]
        },
    )
    body: Final = _request_body(route)
    first_content: Final = cast(Mapping[str, JsonValue], cast(list[JsonValue], body["contents"])[0])
    parts: Final = cast(list[JsonValue], first_content["parts"])
    audio_part: Final = cast(Mapping[str, JsonValue], parts[1])
    assert cast(Mapping[str, JsonValue], audio_part["inline_data"])["data"] == wav_b64
    assert response.choices[0].message.content == "canned-gemini"


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_novalite",
        "bedrock_converse_gptoss",
        "bedrock_invoke_haiku",
        "bedrock_invoke_novamicro",
        "gemini_25flash",
        "groq_oss120b",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
        "together_glm",
    ),
    ids=_case_id,
)
def test_json_response_format_stream(case: _Case, respx_mock: MockRouter) -> None:
    canned: Final = '{"city":"San Francisco"}'
    route: Final = _register(case, respx_mock, canned, stream=True)
    response: Final = _stream(
        case,
        {
            "messages": [
                {"role": "system", "content": "Your output should be a JSON object with no additional properties."},
                {"role": "user", "content": "Respond with this in json. city=San Francisco, state=CA"},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    content: Final = "".join(chunk.choices[0].delta.content or "" for chunk in response)
    assert content == canned
    body: Final = _request_body(route)
    if case["shape"] in ("openai", "anthropic") and not case["id"].startswith("groq"):
        assert body["stream"] is True


@pytest.mark.parametrize(
    "case",
    _pick(
        "bedrock_invoke_haiku",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "router_gpt4omini",
        "together_glm",
    ),
    ids=_case_id,
)
@pytest.mark.parametrize("detail", (None, "low", "high"), ids=("detail_none", "detail_low", "detail_high"))
@pytest.mark.parametrize("image_url", _PNG_URLS, ids=("litellm_logo", "awsmp_png"))
def test_image_url(case: _Case, detail: str | None, image_url: str, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    image_url_part: Final[Mapping[str, JsonValue]] = (
        {"url": image_url} if detail is None else {"url": image_url, "detail": detail}
    )
    response: Final = _complete(
        case,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": image_url_part},
                    ],
                }
            ]
        },
    )
    body: Final = _request_body(route)
    assert "What's in this image?" in _user_texts(case, body)
    content: Final = _mappings(_mappings(body["messages"])[0]["content"])
    image_block: Final = content[1]
    if case["shape"] == "bedrock_invoke":
        assert image_block["type"] == "image"
        source: Final = _mapping(image_block["source"])
        assert source["type"] == "base64"
        assert source["data"] == _PNG_B64
        assert source["media_type"] == ("image/jpeg" if image_url.endswith(".jpg") else "image/png")
    else:
        assert image_block["type"] == "image_url"
        assert image_block["image_url"] == image_url_part
    assert response.choices[0].message.content == f"canned-{case['id']}"


@pytest.mark.parametrize(
    "case",
    _pick(
        "bedrock_converse_haiku",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "gemini_25flash",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "router_gpt4omini",
        "together_glm",
    ),
    ids=_case_id,
)
def test_image_url_string(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(
        case,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": _PNG_URLS[1]},
                    ],
                }
            ]
        },
    )
    body: Final = _request_body(route)
    assert "What's in this image?" in _user_texts(case, body)
    image_block: Final = _items(
        _mappings(body["contents"])[0]["parts"]
        if case["shape"] == "gemini"
        else _mappings(body["messages"])[0]["content"]
    )[1]
    match case["shape"]:
        case "openai":
            assert image_block == {"type": "image_url", "image_url": {"url": _PNG_URLS[1]}}
        case _:
            assert _PNG_B64 in json.dumps(image_block)
            assert _PNG_URLS[1] not in json.dumps(image_block)
    assert response.choices[0].message.content == f"canned-{case['id']}"


@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "bedrock_converse_haiku_xregion",
        "bedrock_converse_novalite",
        "bedrock_converse_gptoss",
        "bedrock_invoke_haiku",
        "bedrock_invoke_kimi",
        "gemini_25flash",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o3mini",
        "router_gpt4omini",
    ),
    ids=_case_id,
)
def test_empty_tools(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = _complete(
        case,
        {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "tools": [],
        },
    )
    body: Final = _request_body(route)
    if case["shape"] in ("bedrock_converse", "bedrock_invoke_nova", "gemini"):
        assert "toolConfig" not in body
        assert "tools" not in body
    else:
        assert _tools_payload(case, body) == []
    assert response.choices[0].message.content == f"canned-{case['id']}"


def _cost_model_key(case: _Case) -> str | None:
    model: Final = cast(str, case["kwargs"]["model"])
    stripped: Final = model.split("/", 1)[-1]
    for candidate in (stripped, model):
        if candidate in litellm.model_cost:
            return candidate
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    _pick(
        "anthropic_sonnet45",
        "azure_o3mini",
        "bedrock_converse_haiku",
        "bedrock_converse_novalite",
        "bedrock_converse_novamicro",
        "bedrock_converse_llama33",
        "bedrock_invoke_haiku",
        "gemini_25flash",
        "groq_oss120b",
        "mistral_medium",
        "openai_gpt4omini",
        "openai_o1",
        "openai_o3mini",
        "router_gpt4omini",
        "together_glm",
        "xai_grok3mini",
    ),
    ids=_case_id,
)
async def test_completion_cost(case: _Case, respx_mock: MockRouter) -> None:
    _register(case, respx_mock, f"canned-{case['id']}")
    response: Final = await _acomplete(
        case,
        {"messages": [{"role": "user", "content": "Hello, how are you?"}]},
    )
    usage: Final = response.usage
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    model_key: Final = _cost_model_key(case)
    cost_entry: Final = litellm.model_cost.get(model_key) if model_key is not None else None
    actual_cost: Final = response._hidden_params["response_cost"]
    if cost_entry is not None and "input_cost_per_token" in cost_entry:
        expected: Final = 10 * cost_entry["input_cost_per_token"] + 5 * cost_entry["output_cost_per_token"]
        assert actual_cost == pytest.approx(expected)
    else:
        try:
            expected_cost: Final = litellm.completion_cost(
                completion_response=response, model=cast(str, case["kwargs"]["model"])
            )
        except litellm.exceptions.ModelNotMappedError:
            assert actual_cost is None
        else:
            assert actual_cost == expected_cost


@pytest.mark.parametrize("input_type", ("input_audio", "audio_url"))
def test_supports_audio_input_gemini(input_type: str) -> None:
    wav_b64: Final = base64.b64encode(b"RIFFFAKEWAVDATA").decode()
    audio_part: Final[Mapping[str, JsonValue]] = (
        {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}
        if input_type == "input_audio"
        else {
            "type": "file",
            "file": {"file_id": "gs://bucket/file.wav", "filename": "my-sample-audio-file"},
        }
    )
    raw_request: Final = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": "gemini/gemini-2.5-flash",
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
            "drop_params": True,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is in this recording?"}, audio_part],
                }
            ],
        },
    )
    assert raw_request.get("error") is None
    serialized: Final = json.dumps(raw_request.get("raw_request_body"))
    if input_type == "input_audio":
        assert wav_b64 in serialized
    else:
        assert "gs://bucket/file.wav" in serialized


def test_reasoning_effort_gemini(respx_mock: MockRouter) -> None:
    case: Final = _BY_ID["gemini_25flash"]
    route: Final = _register(case, respx_mock, "canned-gemini")
    optional_params: Final = litellm.get_optional_params(
        model="gemini/gemini-2.5-flash",
        custom_llm_provider="gemini",
        reasoning_effort="high",
    )
    assert optional_params["thinkingConfig"] == {
        "thinkingBudget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
        "includeThoughts": True,
    }
    response: Final = _complete(
        case,
        {
            "messages": [{"role": "user", "content": "Hello!"}],
            "reasoning_effort": "low",
        },
    )
    body: Final = _request_body(route)
    config: Final = cast(Mapping[str, JsonValue], body["generationConfig"])
    assert config["thinkingConfig"] == {
        "thinkingBudget": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
        "includeThoughts": True,
    }
    assert response.choices[0].message.content == "canned-gemini"


@pytest.mark.parametrize("case", _pick("openai_o1", "openai_o3mini", "azure_o3mini", "azure_o3mini_live"), ids=_case_id)
def test_o_series_reasoning_effort_forwarded(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    _complete(
        case,
        {
            "messages": [{"role": "user", "content": "Hello!"}],
            "reasoning_effort": "low",
        },
    )
    body: Final = _request_body(route)
    assert body["reasoning_effort"] == "low"


@pytest.mark.parametrize("case", _pick("openai_o1", "openai_o3mini", "azure_o3mini", "azure_o3mini_live"), ids=_case_id)
def test_o_series_developer_role_kept(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    _complete(
        case,
        {
            "messages": [
                {"role": "developer", "content": "Be a good bot!"},
                {"role": "user", "content": "Hello!"},
            ]
        },
    )
    body: Final = _request_body(route)
    first: Final = cast(Mapping[str, JsonValue], cast(list[JsonValue], body["messages"])[0])
    assert first["role"] == "developer"
    assert first["content"] == "Be a good bot!"


@pytest.mark.parametrize("case", _pick("openai_o1", "openai_o3mini", "azure_o3mini", "azure_o3mini_live"), ids=_case_id)
def test_o_series_temperature_dropped(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = _register(case, respx_mock, f"canned-{case['id']}")
    _complete(
        case,
        {
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "temperature": 0.0,
            "drop_params": True,
        },
    )
    body: Final = _request_body(route)
    assert "temperature" not in body
