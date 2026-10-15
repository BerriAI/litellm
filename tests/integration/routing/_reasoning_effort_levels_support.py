from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import yaml
from integration._support.anthropic_sse import message_json, message_stream, stream_reply
from integration._support.client import Gateway, Scenario, object_value
from integration._support.openai_wire import chat_reply, openai_error, responses_reply
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
KIMI_LEVELS: Final = ("low", "high", "max", "none")
CLAUDE_LEVELS: Final = ("low", "high")
GPT_LEVELS: Final = ("low", "medium", "high", "xhigh")
KIMI: Final = "kimi-k3"
CLAUDE: Final = "claude-opus-5-5"
GPT: Final = "gpt-6.1-sol"
GLM: Final = "zai-glm-5-3"
ECHOED_KEY_MODEL: Final = f"{KIMI}-echoed-key"
BEDROCK_CLAUDE: Final = f"us.anthropic.{CLAUDE}"
AZURE_API_VERSION: Final = "2025-04-01-preview"
SYNTHETIC_KEY: Final = "synthetic-provider-key"
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_SCRIPTED_STATUS: Final = re.compile(r"status=(\d{3})")


def new_marker() -> str:
    return uuid.uuid4().hex


def prompt(marker: str) -> str:
    return f"marker-{marker}"


def answer(marker: str) -> str:
    return f"answer marker-{marker}"


def marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode(errors="replace"))
    assert found is not None, request.body[:400]
    return found.group(1)


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def path_of(request: Request) -> str:
    return urlsplit(request.target).path


def _converse_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": answer(marker)}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def provider_reply(request: Request) -> Reply:
    marker: Final = marker_of(request)
    scripted: Final = _SCRIPTED_STATUS.search(request.body.decode(errors="replace"))
    if scripted is not None:
        return openai_error(int(scripted.group(1)))
    body: Final = body_of(request)
    path: Final = path_of(request)
    model: Final = str(body.get("model", ""))
    stream: Final = body.get("stream") is True
    if path.endswith("/converse"):
        return _converse_reply(marker)
    if path.endswith("/invoke"):
        return Reply(body=message_json(f"msg_{marker}", model, answer(marker)))
    if path.endswith("/messages"):
        if stream:
            return stream_reply(message_stream(f"msg_{marker}", model, answer(marker)))
        return Reply(body=message_json(f"msg_{marker}", model, answer(marker)))
    if path.endswith("/responses"):
        return responses_reply(f"resp_{marker}", model, answer(marker), stream=stream)
    if path.endswith("/chat/completions"):
        return chat_reply(f"chatcmpl-{marker}", model, answer(marker), stream=stream)
    return Reply(status=404, body=json.dumps({"error": {"message": f"no scripted route {path}"}}).encode())


@dataclass(frozen=True, slots=True)
class Backend:
    provider_model: str
    params: Mapping[str, JsonValue]


def backend(kind: str, wire_url: str) -> Backend:
    match kind:
        case "azure_ai":
            return Backend(f"azure_ai/{KIMI}", MappingProxyType({"api_base": wire_url, "api_key": SYNTHETIC_KEY}))
        case "azure_ai_claude":
            return Backend(f"azure_ai/{CLAUDE}", MappingProxyType({"api_base": wire_url, "api_key": SYNTHETIC_KEY}))
        case "hosted_vllm":
            return Backend(
                f"hosted_vllm/{KIMI}", MappingProxyType({"api_base": f"{wire_url}/v1", "api_key": SYNTHETIC_KEY})
            )
        case "hosted_vllm_echoed_key":
            return Backend(
                f"hosted_vllm/{ECHOED_KEY_MODEL}",
                MappingProxyType({"api_base": f"{wire_url}/v1", "api_key": SYNTHETIC_KEY}),
            )
        case "hosted_vllm_claude":
            return Backend(
                f"hosted_vllm/{CLAUDE}", MappingProxyType({"api_base": f"{wire_url}/v1", "api_key": SYNTHETIC_KEY})
            )
        case "mistral":
            return Backend(f"mistral/{GLM}", MappingProxyType({"api_base": f"{wire_url}/v1", "api_key": SYNTHETIC_KEY}))
        case "openai":
            return Backend(f"openai/{GPT}", MappingProxyType({"api_base": f"{wire_url}/v1", "api_key": SYNTHETIC_KEY}))
        case "azure":
            return Backend(
                f"azure/{GPT}",
                MappingProxyType({"api_base": wire_url, "api_key": SYNTHETIC_KEY, "api_version": AZURE_API_VERSION}),
            )
        case "anthropic":
            return Backend(f"anthropic/{CLAUDE}", MappingProxyType({"api_base": wire_url, "api_key": SYNTHETIC_KEY}))
        case "bedrock_converse":
            return Backend(
                f"bedrock/converse/{BEDROCK_CLAUDE}",
                MappingProxyType({"api_base": wire_url, "api_key": SYNTHETIC_KEY, "aws_region_name": "us-east-1"}),
            )
        case "bedrock_invoke":
            return Backend(
                f"bedrock/invoke/{BEDROCK_CLAUDE}",
                MappingProxyType({"api_base": wire_url, "api_key": SYNTHETIC_KEY, "aws_region_name": "us-east-1"}),
            )
        case _:
            raise AssertionError(f"unknown backend kind {kind}")


def deployment(scenario: Scenario, kind: str, wire: Wire, model_info: Mapping[str, JsonValue]) -> str:
    chosen: Final = backend(kind, wire.url)
    return scenario.model(model=chosen.provider_model, model_info=model_info, **chosen.params)


def yaml_deployment(
    model_name: str, kind: str, wire_url: str, model_info: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    chosen: Final = backend(kind, wire_url)
    return {
        "model_name": model_name,
        "litellm_params": {"model": chosen.provider_model, **chosen.params},
        "model_info": dict(model_info),
    }


def owned_config(
    directory: Path,
    model_list: Sequence[JsonValue],
    *,
    litellm_settings: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    merged: Final = {
        **config,
        "model_list": list(model_list),
        "litellm_settings": {**object_value(config["litellm_settings"]), "cache": False, **litellm_settings},
        "router_settings": {**object_value(config["router_settings"]), "num_retries": 0},
    }
    path: Final = directory / f"reasoning-effort-levels-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path


def chat_messages(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": prompt(marker)}]


def chat_request(model: str, marker: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": chat_messages(marker), **fields}


def messages_request(model: str, marker: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 4096, "messages": chat_messages(marker), **fields}


def responses_request(model: str, marker: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": prompt(marker), **fields}


def only_request(wire: Wire, marker: str) -> Request:
    received: Final = tuple(request for request in wire.drain() if marker_of(request) == marker)
    assert len(received) == 1, [(request.target, request.body[:300]) for request in received]
    return received[0]


def sent_effort(request: Request) -> JsonValue:
    body: Final = body_of(request)
    path: Final = path_of(request)
    if path.endswith("/converse"):
        fields: Final = body.get("additionalModelRequestFields")
        config: Final = fields.get("output_config") if isinstance(fields, dict) else None
        return config.get("effort") if isinstance(config, dict) else None
    if path.endswith(("/messages", "/invoke")):
        output_config: Final = body.get("output_config")
        return output_config.get("effort") if isinstance(output_config, dict) else None
    if path.endswith("/responses"):
        reasoning: Final = body.get("reasoning")
        return reasoning.get("effort") if isinstance(reasoning, dict) else None
    return body.get("reasoning_effort")


def gateway_post(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[int, str]:
    response: Final = gateway.request("POST", path, body)
    return response.status_code, response.text
