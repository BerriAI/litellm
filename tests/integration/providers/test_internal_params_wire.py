import asyncio
import base64
import json
import os
import struct
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, get_args
from urllib.parse import parse_qs

import openai
import pytest
from integration._support.upstream import INTERNAL_FIELDS
from integration._support.wire import Reply, Request, wire_server
from pydantic import TypeAdapter

import litellm
from litellm.constants import BEDROCK_EMBEDDING_PROVIDERS_LITERAL, BEDROCK_INVOKE_PROVIDERS_LITERAL
from litellm.exceptions import LiteLLMUnknownProvider
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockModelInfo, BedrockRoute
from litellm.types import litellm_params
from litellm.types.litellm_params import OWNED_KWARG_NAMES, wire_names
from litellm.types.utils import LlmProviders
from tests._support.stream_chunk_size import keys_at_every_depth

TEXT: Final = "wire control"
OPENAI_RESPONSE: Final = {
    "id": "chatcmpl-wire",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4.1-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": TEXT}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
}
ANTHROPIC_RESPONSE: Final = {
    "id": "msg_wire",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": TEXT}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 4},
}
GEMINI_RESPONSE: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": TEXT}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
}
CONVERSE_RESPONSE: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": TEXT}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
    "metrics": {"latencyMs": 1},
}
OPENAI_STREAM_CHUNKS: Final = (
    {
        "id": "chatcmpl-wire",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": TEXT}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-wire",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
)
ANTHROPIC_STREAM_EVENTS: Final = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_wire",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": TEXT}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
    {"type": "message_stop"},
)
GEMINI_STREAM_CHUNKS: Final = (
    {"candidates": [{"content": {"role": "model", "parts": [{"text": TEXT}]}, "index": 0}]},
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
    },
)
CONVERSE_STREAM_EVENTS: Final = (
    ("contentBlockDelta", {"delta": {"text": TEXT}, "contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14}, "metrics": {"latencyMs": 1}}),
)
NON_STREAM_BODIES: Final = {
    "openai": OPENAI_RESPONSE,
    "azure": OPENAI_RESPONSE,
    "anthropic": ANTHROPIC_RESPONSE,
    "gemini": GEMINI_RESPONSE,
    "converse": CONVERSE_RESPONSE,
    "invoke": ANTHROPIC_RESPONSE,
}
PROVIDERS: Final = ("openai", "azure", "anthropic", "gemini", "converse", "invoke")


def _aws_string_header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return struct.pack("!B", len(name_bytes)) + name_bytes + b"\x07" + struct.pack("!H", len(value_bytes)) + value_bytes


def _aws_event_frame(event_type: str, payload: Mapping[str, object]) -> bytes:
    body: Final = json.dumps(payload, separators=(",", ":")).encode()
    headers: Final = (
        _aws_string_header(":event-type", event_type)
        + _aws_string_header(":content-type", "application/json")
        + _aws_string_header(":message-type", "event")
    )
    prelude: Final = struct.pack("!II", 12 + len(headers) + len(body) + 4, len(headers))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF) + headers + body
    return message + struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)


def _sse_reply(frames: tuple[bytes, ...]) -> Reply:
    return Reply(chunks=frames, content_type="text/event-stream")


def _stream_reply(provider: str) -> Reply:
    match provider:
        case "openai" | "azure":
            return _sse_reply(
                tuple(
                    f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode() for chunk in OPENAI_STREAM_CHUNKS
                )
                + (b"data: [DONE]\n\n",)
            )
        case "anthropic":
            return _sse_reply(
                tuple(
                    f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
                    for event in ANTHROPIC_STREAM_EVENTS
                )
            )
        case "gemini":
            return _sse_reply(
                tuple(
                    f"data: {json.dumps(chunk, separators=(',', ':'))}\r\n\r\n".encode()
                    for chunk in GEMINI_STREAM_CHUNKS
                )
            )
        case "converse":
            return Reply(
                chunks=tuple(_aws_event_frame(event_type, payload) for event_type, payload in CONVERSE_STREAM_EVENTS),
                content_type="application/vnd.amazon.eventstream",
            )
        case "invoke":
            return Reply(
                chunks=tuple(
                    _aws_event_frame(
                        "chunk", {"bytes": base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()}
                    )
                    for event in ANTHROPIC_STREAM_EVENTS
                ),
                content_type="application/vnd.amazon.eventstream",
            )


def _request_parameters(provider: str, wire_url: str) -> dict[str, object]:
    common: Final = {"messages": [{"role": "user", "content": "synthetic chunk control"}]}
    match provider:
        case "openai":
            return {**common, "model": "openai/gpt-4.1-mini", "api_key": "synthetic-openai-key", "api_base": wire_url}
        case "azure":
            return {
                **common,
                "model": "azure/gpt-4.1-mini",
                "api_key": "synthetic-azure-key",
                "api_base": wire_url,
                "api_version": "2025-01-01-preview",
            }
        case "anthropic":
            return {
                **common,
                "model": "anthropic/claude-sonnet-4-5",
                "api_key": "synthetic-anthropic-key",
                "api_base": wire_url,
            }
        case "gemini":
            return {
                **common,
                "model": "gemini/gemini-2.5-flash",
                "api_key": "synthetic-gemini-key",
                "api_base": wire_url,
            }
        case "converse":
            return {
                **common,
                "model": "bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0",
                "aws_access_key_id": "fake",
                "aws_secret_access_key": "fake",
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire_url,
            }
        case "invoke":
            return {
                **common,
                "model": "bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
                "aws_access_key_id": "fake",
                "aws_secret_access_key": "fake",
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire_url,
            }


def _expected_target(provider: str, streaming: bool) -> str:
    match provider:
        case "openai":
            return "/chat/completions"
        case "azure":
            return "/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
        case "anthropic":
            return "/v1/messages"
        case "gemini":
            return ":streamGenerateContent" if streaming else ":generateContent"
        case "converse":
            return "/converse-stream" if streaming else "/converse"
        case "invoke":
            return "/invoke-with-response-stream" if streaming else "/invoke"


def _at(value: object, *path: str) -> object:
    if not path:
        return value
    assert isinstance(value, Mapping)
    return _at(value[path[0]], *path[1:])


def _custom_key(body: Mapping[str, object], provider: str) -> object:
    match provider:
        case "anthropic":
            return _at(body, "extra_body", "custom_provider_key")
        case "converse":
            return _at(body, "additionalModelRequestFields", "extra_body", "custom_provider_key")
    return _at(body, "custom_provider_key")


def _peer(provider: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body) if request.body else {}
        streaming: Final = (
            (isinstance(body, dict) and body.get("stream") is True)
            or "streamGenerateContent" in request.target
            or request.target.endswith(("-stream",))
        )
        expected: Final = _expected_target(provider, streaming)
        assert expected in request.target, f"{provider}: expected {expected} in {request.target}"
        return _stream_reply(provider) if streaming else Reply(body=json.dumps(NON_STREAM_BODIES[provider]).encode())

    return respond


@pytest.fixture
def provider_wire_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty: Final = tmp_path / "empty-aws-config"
    empty.write_text("")
    for name in tuple(name for name in os.environ if name.startswith("AWS_")):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "AWS_CONFIG_FILE": str(empty),
        "AWS_SHARED_CREDENTIALS_FILE": str(empty),
        "AWS_EC2_METADATA_DISABLED": "true",
        "LITELLM_RUST": "false",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_internal_params_never_reach_provider_body(
    provider_wire_environment: None,
    provider: str,
    asynchronous: bool,
    stream: bool,
) -> None:
    with wire_server(_peer(provider)) as wire:
        parameters: Final = {
            **_request_parameters(provider, wire.url),
            "stream": stream,
            "stream_chunk_size": 64,
            "_litellm_undeclared_sentinel": "internal",
            "extra_body": {"custom_provider_key": 1},
            "max_tokens": 16,
            "timeout": 5,
            "num_retries": 0,
        }
        result: Final = (
            await litellm.acompletion(**parameters)
            if asynchronous
            else await asyncio.to_thread(litellm.completion, **parameters)
        )
        if stream:
            chunks: Final = [chunk async for chunk in result] if asynchronous else [chunk for chunk in result]
            text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            assert text == TEXT
        else:
            assert result.choices[0].message.content == TEXT
        requests: Final = wire.drain()
        assert len(requests) == 1
        body: Final = json.loads(requests[0].body)
        keys: Final = keys_at_every_depth(body)
        assert "stream_chunk_size" not in keys
        assert not INTERNAL_FIELDS.intersection(keys)
        assert not frozenset(key for key in keys if key.startswith("_litellm_")), keys
        assert _custom_key(body, provider) == 1


WALK_TEXT: Final = "walk every provider"
UNDECLARED_SENTINEL: Final = "_litellm_undeclared_sentinel"


def _ignore_log_details(_details: Mapping[str, object]) -> None:
    return None


WALK_SEEDS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "aws_batch_role_arn": "arn:aws:iam::000000000000:role/wire",
        "s3_bucket_name": "wire-bucket",
        "s3_region_name": "us-east-1",
        "s3_endpoint_url": "https://s3.wire.invalid",
        "s3_output_bucket_name": "wire-output",
        "s3_bucket_owner": "000000000000",
        "s3_access_key_id": "wire-access-key",
        "s3_secret_access_key": "wire-secret-key",
        "s3_encryption_key_id": "wire-kms-key",
        "bedrock_tags": [{"key": "wire", "value": "walk"}],
        "model_info": {"id": "wire-deployment"},
        "rpm": 7,
        "tpm": 7,
        "itpm": 7,
        "otpm": 7,
        "default_api_key_rpm_limit": 7,
        "default_api_key_tpm_limit": 7,
        "max_parallel_requests": 1,
        "weight": 1,
        "order": 1,
        "tag_regex": ["^wire$"],
        "max_file_size_mb": 1.0,
        "auto_router_config_path": "wire-router.json",
        "auto_router_config": "wire-router",
        "auto_router_default_model": "wire-default",
        "auto_router_embedding_model": "wire-embedding",
        "auto_router_max_input_chars": 64,
        "auto_router_routing_compression": "wire",
        "auto_router_model_compression": "wire",
        "complexity_router_config": {"wire": 1},
        "complexity_router_default_model": "wire-default",
        "adaptive_router_config": {"wire": 1},
        "adaptive_router_default_model": "wire-default",
        "quality_router_config": {"wire": 1},
        "quality_router_default_model": "wire-default",
        "cost_per_query": 0.5,
        "max_budget": 100.0,
        "budget_duration": "30d",
        "id": "wire-id",
        "metadata": {"wire": 1},
        "litellm_metadata": {"wire": 1},
        "tags": ["wire"],
        "litellm_trace_id": "wire-trace",
        "litellm_session_id": "wire-session",
        "litellm_request_debug": False,
        "logger_fn": _ignore_log_details,
        "verbose": False,
        "no-log": False,
        "log_client_error_tracebacks": False,
        "max_agentic_loops": 1,
        "merge_reasoning_content_in_choices": False,
        "enable_json_schema_validation": False,
        "complete_response": False,
        "keepalive_seconds": 30.0,
        "allow_client_keepalive_override": False,
        "stream_chunk_size": 64,
        "litellm_call_id": "wire-call",
        "completion_call_id": "wire-completion",
        "preset_cache_key": "wire-cache-key",
        "cache_key": "wire-cache-key",
        "fake_stream": False,
        "_skip_mcp_handler": True,
        "skip_mcp_handler": True,
        "mcp_handler_context": {"server": "wire"},
        "fallback_depth": 1,
        "max_fallbacks": 1,
        UNDECLARED_SENTINEL: "internal",
    }
)
OWNED_LEAVES_NOT_SEEDED: Final = MappingProxyType(
    {
        litellm_params.ProviderConnection: "the route needs its own connection settings to send anything",
        litellm_params.AnthropicFederationConnection: "switches the route to workload identity federation",
        litellm_params.OpenAIFederationConnection: "switches the route to workload identity federation",
        litellm_params.DispatchOptions: "chooses which provider route handles the call",
        litellm_params.RoutingOptions: "retries and fallbacks change which requests are sent",
        litellm_params.CachingOptions: "turns response caching on or off",
        litellm_params.GuardrailOptions: "runs the named guardrails before the call",
        litellm_params.PromptOptions: "changes how the prompt is built or loads a managed prompt",
        litellm_params.MockOptions: "answers without sending a request",
        litellm_params.AgenticLoopState: "drives the agentic loop between follow-up calls",
        litellm_params.ProxyRequestState: "set by the proxy for its own request handling",
        litellm_params.EntrypointState: "picks the sync or async entry point",
    }
)
OWNED_NAMES_NOT_SEEDED: Final = MappingProxyType(
    {
        "silent_model": "sends a second request to another model",
        "base_model": "stands in for the model name when a provider config resolves the model",
        "model_alias_map": "renames the model before the route is chosen",
        "data_residency": "picks a regional endpoint",
        "litellm_logging_obj": "must be the call's own logging object",
        "stream_response": "holds a response object, not request input",
        "_context_compaction_state": "holds compaction state, not request input",
        "_router_weights": "holds router weight state, not request input",
        "attempted_targets": "holds the router's fallback bookkeeping, not request input",
    }
)
EMBEDDING_WALK_SEEDS: Final[Mapping[str, object]] = MappingProxyType(
    {
        **WALK_SEEDS,
        "cache_control_injection_points": [{"location": "message", "role": "user"}],
        "knowledge_bases": ["kb_wire"],
        "vector_store_ids": ["vs_wire"],
    }
)
WALK_REJECTION: Final = Reply(
    status=400,
    body=json.dumps({"error": {"message": "rejected by the wire", "type": "invalid_request_error"}}).encode(),
)
SYNTHETIC_KEY: Final = MappingProxyType({"api_key": "synthetic-key"})
AZURE_CONNECTION: Final = MappingProxyType({**SYNTHETIC_KEY, "api_version": "2025-01-01-preview"})
AWS_CONNECTION: Final = MappingProxyType(
    {"aws_access_key_id": "fake", "aws_secret_access_key": "fake", "aws_region_name": "us-east-1"}
)
AWS_BASE_PARAMS: Final = ("api_base", "aws_bedrock_runtime_endpoint")
ONE_PIXEL_GIF: Final = "data:image/gif;base64,R0lGODlhAQABAAAAACw="


@dataclass(frozen=True, slots=True)
class Route:
    model: str
    connection: Mapping[str, object] = SYNTHETIC_KEY
    base_params: tuple[str, ...] = ("api_base",)
    base_suffix: str = ""
    request: Mapping[str, object] = MappingProxyType({})
    protocol_fields: frozenset[str] = frozenset()
    streams: bool = True

    def parameters(self, wire_url: str) -> dict[str, object]:
        return {
            "model": self.model,
            **self.connection,
            **dict.fromkeys(self.base_params, wire_url + self.base_suffix),
            **self.request,
        }


def _bedrock(model: str, **connection: object) -> Route:
    return Route(f"bedrock/{model}", MappingProxyType({**AWS_CONNECTION, **connection}), AWS_BASE_PARAMS)


IMPORTED_MODEL: Final = "arn:aws:bedrock:us-east-1:000000000000:imported-model/wire"
BEDROCK_INVOKE_MODELS: Final = MappingProxyType(
    {
        "anthropic": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "cohere": "cohere.command-r-plus-v1:0",
        "mistral": "mistral.mistral-large-2402-v1:0",
        "amazon": "amazon.titan-text-express-v1",
        "meta": "meta.llama3-1-70b-instruct-v1:0",
        "llama": f"llama/{IMPORTED_MODEL}",
        "ai21": "ai21.jamba-1-5-large-v1:0",
        "nova": "amazon.nova-pro-v1:0",
        "deepseek_r1": f"deepseek_r1/{IMPORTED_MODEL}",
        "qwen3": f"qwen3/{IMPORTED_MODEL}",
        "qwen2": f"qwen2/{IMPORTED_MODEL}",
        "twelvelabs": "twelvelabs.pegasus-1-2-v1:0",
        "openai": "openai.gpt-oss-120b-1:0",
        "moonshot": "moonshot.kimi-k2-thinking",
    }
)
BEDROCK_INVOKE_FAMILIES_WITHOUT_CHAT: Final = MappingProxyType({"stability": "generates images only"})
BEDROCK_ROUTES: Final = MappingProxyType(
    {
        "converse": _bedrock("converse/anthropic.claude-haiku-4-5-20251001-v1:0"),
        "converse_like": _bedrock("converse_like/anthropic.claude-haiku-4-5-20251001-v1:0"),
        "openai": _bedrock("openai/wire-model"),
        "chat_completions": _bedrock("chat_completions/openai.gpt-oss-120b-1:0"),
        "mantle": _bedrock("mantle/openai.gpt-oss-120b"),
        "claude_platform": _bedrock("claude_platform/claude-haiku-4-5", workspace_id="wrk_wire"),
        "agent": _bedrock("agent/AGENTID/ALIASID"),
        "agentcore": _bedrock("agentcore/arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/wire"),
        **{f"invoke/{family}": _bedrock(f"invoke/{model}") for family, model in BEDROCK_INVOKE_MODELS.items()},
    }
)
BEDROCK_ROUTES_WITHOUT_CHAT: Final = MappingProxyType({"async_invoke": "starts an async job instead of a chat call"})

API_BASE_CHAT_PROVIDERS: Final = (
    "ai21", "ai21_chat", "aiml", "aiohttp_openai", "amazon_nova", "anthropic", "anthropic_text", "apertis",
    "assemblyai", "azure_ai", "baseten", "cerebras", "charity_engine", "chutes", "cloudflare", "codestral",
    "cognition", "cohere", "cohere_chat", "cometapi", "compactifai", "cortecs", "custom", "custom_openai",
    "darkbloom", "dashscope", "databricks", "datarobot", "deepinfra", "deepseek", "docker_model_runner", "edenai",
    "empower", "featherless_ai", "fireworks_ai", "friendliai", "galadriel", "gemini", "github", "groq", "helicone",
    "heroku", "hosted_vllm", "huggingface", "hyperbolic", "inception", "lambda_ai", "langflow", "langgraph",
    "lemonade", "libertai", "litellm_proxy", "llamafile", "lm_studio", "maritalk", "meta", "meta_llama", "minimax",
    "mistral", "modelscope", "moonshot", "morph", "nadir", "nano-gpt", "nebius", "neosantara", "novita", "nscale",
    "nvidia_nim", "ollama", "ollama_chat", "oobabooga", "openai", "openrouter", "ovhcloud", "parasail", "perplexity",
    "petals", "pinstripes", "poe", "prism", "publicai", "qwen_ai_platform", "qwencloud", "reka", "sail", "sambanova",
    "scaleway", "scx-ai", "snowflake", "synthetic", "tencent", "tensormesh", "text-completion-codestral",
    "text-completion-inception", "text-completion-openai", "together_ai", "v0", "vercel_ai_gateway", "volcengine",
    "wandb", "xai", "xiaomi_mimo", "xinference", "zai",
)  # fmt: skip
CHAT_ROUTES: Final = MappingProxyType(
    {
        **{provider: Route(f"{provider}/wire-model") for provider in API_BASE_CHAT_PROVIDERS},
        "a2a": Route("a2a/wire-model", protocol_fields=frozenset({"id"})),
        "azure": Route("azure/wire-model", AZURE_CONNECTION),
        "azure_text": Route("azure_text/wire-model", AZURE_CONNECTION),
        "bedrock_mantle": Route("bedrock_mantle/wire-model", AWS_CONNECTION, AWS_BASE_PARAMS),
        "clarifai": Route("clarifai/user.app.wire-model"),
        "fal_ai": Route(
            "fal_ai/wire-model",
            request=MappingProxyType(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": WALK_TEXT},
                                {"type": "image_url", "image_url": {"url": ONE_PIXEL_GIF}},
                            ],
                        }
                    ]
                }
            ),
            streams=False,
        ),
        "gradient_ai": Route("gradient_ai/wire-model"),
        "nlp_cloud": Route("nlp_cloud/chatdolphin", base_suffix="/"),
        "predibase": Route("predibase/wire-model", MappingProxyType({**SYNTHETIC_KEY, "predibase_tenant_id": "wire"})),
        "ragflow": Route("ragflow/chat/wire-chat/wire-model"),
        "scaledown": Route("scaledown/summarize"),
        "triton": Route("triton/wire-model", base_suffix="/generate"),
        **{f"bedrock/{route}": target for route, target in BEDROCK_ROUTES.items()},
    }
)

VENDOR_HOST: Final = "calls the vendor host even when api_base is set"
VENDOR_AUTH: Final = "needs vendor credentials or a token exchange the wire cannot fake"
DEVICE_LOGIN: Final = "needs an interactive device-code login"
CHAT_PROVIDERS_OFF_THE_WIRE: Final = MappingProxyType(
    {
        **dict.fromkeys(("bytez", "replicate", "sagemaker", "sagemaker_chat", "sagemaker_nova"), VENDOR_HOST),
        **dict.fromkeys(
            ("gdc", "gigachat", "microsoft_365_copilot", "oci", "sap", "vertex_ai", "vertex_ai_beta", "watsonx",
             "watsonx_text"),
            VENDOR_AUTH,
        ),
        **dict.fromkeys(("chatgpt", "github_copilot"), DEVICE_LOGIN),
        "vllm": "runs the local vllm package instead of calling a server",
    }
)  # fmt: skip

EMBEDDING_ROUTES: Final = MappingProxyType(
    {
        **{
            provider: Route(f"{provider}/wire-model")
            for provider in (
                "azure_ai", "cohere", "cohere_chat", "cometapi", "dashscope", "databricks", "edenai", "fireworks_ai",
                "gemini", "hosted_vllm", "infinity", "jina_ai", "litellm_proxy", "llamafile", "lm_studio", "mistral",
                "nebius", "nvidia_nim", "ollama", "oobabooga", "openai", "openai_like", "openrouter", "ovhcloud",
                "perplexity", "qwen_ai_platform", "qwencloud", "sambanova", "snowflake", "together_ai", "triton",
                "vercel_ai_gateway", "volcengine", "voyage", "wandb", "xinference",
            )
        },
        "azure": Route("azure/wire-model", AZURE_CONNECTION),
        "huggingface": Route("huggingface/wire-model", request=MappingProxyType({"input_type": "feature-extraction"})),
        "bedrock/amazon": _bedrock("amazon.titan-embed-image-v1"),
        "bedrock/cohere": _bedrock("cohere.embed-english-v3"),
        "bedrock/nova": _bedrock("amazon.nova-2-multimodal-embeddings-v1:0"),
        "bedrock/twelvelabs": _bedrock("twelvelabs.marengo-embed-2-7-v1:0"),
    }
)  # fmt: skip
EMBEDDING_PROVIDERS_OFF_THE_WIRE: Final = MappingProxyType(
    {
        "sagemaker": VENDOR_HOST,
        **dict.fromkeys(("gigachat", "oci", "sap", "vertex_ai", "watsonx"), VENDOR_AUTH),
        **dict.fromkeys(("chatgpt", "github_copilot"), DEVICE_LOGIN),
    }
)
REFUSALS: Final = (openai.OpenAIError, BaseLLMException, ValueError)

CHAT_ROUTE_BUGS: Final = MappingProxyType(
    {
        "gradient_ai": "BUG: rejects the max_retries that completion() derives from num_retries, before sending",
        "huggingface": "BUG: sends the max_retries that completion() derives from num_retries in the request body",
        "ragflow": "BUG: sends the max_retries that completion() derives from num_retries in the request body",
    }
)
JSON_BODY: Final = TypeAdapter(dict[str, object])


def _provider(route_id: str) -> str:
    return route_id.split("/", 1)[0]


def _chat_cases() -> Iterator[tuple[str, bool]]:
    for route_id, route in CHAT_ROUTES.items():
        yield route_id, False
        if route.streams:
            yield route_id, True


def _unwalked(routes: Mapping[str, Route], off_the_wire: Mapping[str, str]) -> tuple[str, ...]:
    walked: Final = frozenset(_provider(route_id) for route_id in routes)
    return tuple(provider.value for provider in LlmProviders if provider.value not in walked | frozenset(off_the_wire))


def _unrouted_model(provider: str) -> str:
    return CHAT_ROUTES[provider].model if provider in CHAT_ROUTES else f"{provider}/wire-model"


def _complete(parameters: Mapping[str, object]) -> object:
    return litellm.completion(**{"messages": [{"role": "user", "content": WALK_TEXT}], **parameters})


def _embed(parameters: Mapping[str, object]) -> object:
    return litellm.embedding(**{"input": [WALK_TEXT], **parameters})


def _unrouted_parameters(provider: str, wire_url: str) -> dict[str, object]:
    return {"model": _unrouted_model(provider), **SYNTHETIC_KEY, "api_base": wire_url, "num_retries": 0, "timeout": 5}


def _sent_body(request: Request) -> Mapping[str, object]:
    if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
        fields: Final = parse_qs(request.body.decode(), keep_blank_values=True, strict_parsing=True)
        return {name: values[0] for name, values in fields.items()}
    return JSON_BODY.validate_json(request.body)


def _walked_bodies(
    route: Route, seeds: Mapping[str, object], send: Callable[[dict[str, object]], object]
) -> tuple[Mapping[str, object], ...]:
    with wire_server(lambda _request: WALK_REJECTION) as wire:
        with suppress(openai.OpenAIError, BaseLLMException):
            result: Final = send({**seeds, **route.parameters(wire.url), "num_retries": 0, "timeout": 5})
            if isinstance(result, Iterator):
                for _chunk in result:
                    pass
        requests: Final = wire.drain()
    assert requests, f"{route.model} sent nothing to the wire"
    return tuple(_sent_body(request) for request in requests if request.body)


def _assert_only_provider_fields(
    bodies: tuple[Mapping[str, object], ...], seeds: Mapping[str, object], route: Route
) -> None:
    assert any(WALK_TEXT in json.dumps(body) for body in bodies), bodies
    for body in bodies:
        assert not (frozenset(seeds) - route.protocol_fields) & keys_at_every_depth(body), body
        assert not (frozenset(OWNED_KWARG_NAMES) - route.protocol_fields) & frozenset(body), body
        assert not tuple(key for key in body if key.startswith("_")), body


def test_the_form_body_parser_keeps_a_blank_leaked_field() -> None:
    form: Final = Request(
        method="POST",
        target="/",
        headers={"content-type": "application/x-www-form-urlencoded"},
        body=b"inputs=walk+every+provider&skip_mcp_handler=",
    )

    with pytest.raises(AssertionError):
        _assert_only_provider_fields((_sent_body(form),), WALK_SEEDS, Route("petals/wire-model"))


def test_every_owned_name_is_seeded_or_has_a_reason_it_is_not() -> None:
    seeded: Final = frozenset(WALK_SEEDS) - {UNDECLARED_SENTINEL}
    unseeded: Final = frozenset(OWNED_NAMES_NOT_SEEDED).union(*map(wire_names, OWNED_LEAVES_NOT_SEEDED))

    assert seeded.isdisjoint(unseeded), seeded & unseeded
    assert seeded | unseeded == frozenset(OWNED_KWARG_NAMES), frozenset(OWNED_KWARG_NAMES) ^ (seeded | unseeded)


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize(("route_id", "stream"), tuple(_chat_cases()))
def test_every_chat_route_keeps_internal_params_out_of_the_provider_body(
    provider_wire_environment: None, route_id: str, stream: bool
) -> None:
    if route_id in CHAT_ROUTE_BUGS:
        pytest.skip(CHAT_ROUTE_BUGS[route_id])

    bodies: Final = _walked_bodies(
        CHAT_ROUTES[route_id], WALK_SEEDS, lambda parameters: _complete({**parameters, "stream": stream})
    )

    _assert_only_provider_fields(bodies, WALK_SEEDS, CHAT_ROUTES[route_id])


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize("route_id", EMBEDDING_ROUTES)
def test_every_embedding_route_keeps_internal_params_out_of_the_provider_body(
    provider_wire_environment: None, route_id: str
) -> None:
    bodies: Final = _walked_bodies(EMBEDDING_ROUTES[route_id], EMBEDDING_WALK_SEEDS, _embed)

    _assert_only_provider_fields(bodies, EMBEDDING_WALK_SEEDS, EMBEDDING_ROUTES[route_id])


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize("provider", CHAT_PROVIDERS_OFF_THE_WIRE)
def test_a_chat_provider_kept_off_the_wire_sends_nothing_to_it(provider_wire_environment: None, provider: str) -> None:
    with wire_server(lambda _request: WALK_REJECTION) as wire:
        with suppress(*REFUSALS):
            _complete(_unrouted_parameters(provider, wire.url))
        sent: Final = wire.drain()

    assert not sent, f"{provider} reaches the wire, so walk it in CHAT_ROUTES instead"


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize("provider", _unwalked(CHAT_ROUTES, CHAT_PROVIDERS_OFF_THE_WIRE))
def test_a_provider_without_a_chat_route_has_no_chat_handler(provider_wire_environment: None, provider: str) -> None:
    with wire_server(lambda _request: WALK_REJECTION) as wire:
        with pytest.raises(LiteLLMUnknownProvider):
            _complete(_unrouted_parameters(provider, wire.url))
        sent: Final = wire.drain()

    assert not sent, sent


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize("provider", EMBEDDING_PROVIDERS_OFF_THE_WIRE)
def test_an_embedding_provider_kept_off_the_wire_sends_nothing_to_it(
    provider_wire_environment: None, provider: str
) -> None:
    with wire_server(lambda _request: WALK_REJECTION) as wire:
        with suppress(*REFUSALS):
            _embed(_unrouted_parameters(provider, wire.url))
        sent: Final = wire.drain()

    assert not sent, f"{provider} reaches the wire, so walk it in EMBEDDING_ROUTES instead"


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize("provider", _unwalked(EMBEDDING_ROUTES, EMBEDDING_PROVIDERS_OFF_THE_WIRE))
def test_a_provider_without_an_embedding_route_has_no_embedding_handler(
    provider_wire_environment: None, provider: str
) -> None:
    with wire_server(lambda _request: WALK_REJECTION) as wire:
        with pytest.raises(LiteLLMUnknownProvider):
            _embed(_unrouted_parameters(provider, wire.url))
        sent: Final = wire.drain()

    assert not sent, sent


def test_every_bedrock_route_and_invoke_family_is_walked() -> None:
    routes: Final = {
        route: BedrockModelInfo.get_bedrock_route(target.model) for route, target in BEDROCK_ROUTES.items()
    }
    families: Final = {
        family: BaseAWSLLM.get_bedrock_invoke_provider(model=f"invoke/{model}")
        for family, model in BEDROCK_INVOKE_MODELS.items()
    }

    assert routes == {route: route.split("/", 1)[0] for route in BEDROCK_ROUTES}
    assert families == {family: family for family in BEDROCK_INVOKE_MODELS}
    assert {*routes.values(), *BEDROCK_ROUTES_WITHOUT_CHAT} == frozenset(get_args(BedrockRoute))
    assert {*families, *BEDROCK_INVOKE_FAMILIES_WITHOUT_CHAT} == frozenset(get_args(BEDROCK_INVOKE_PROVIDERS_LITERAL))
    assert {route_id.split("/", 1)[1] for route_id in EMBEDDING_ROUTES if route_id.startswith("bedrock/")} == frozenset(
        get_args(BEDROCK_EMBEDDING_PROVIDERS_LITERAL)
    )
