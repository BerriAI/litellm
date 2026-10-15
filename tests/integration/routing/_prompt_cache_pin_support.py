import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml
from integration._support.client import Gateway, eventually, held, object_value
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire
from integration.providers._cache_control_marks_support import anthropic_peer, bedrock_peer, marker_of
from pydantic import JsonValue, TypeAdapter
from redis import Redis

FIVE_MINUTES: Final = 300
ONE_HOUR: Final = 3600
PROVIDER_KEY: Final = "synthetic-provider-key"
BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
LONG_TEXT: Final = "Answer from the reference below. " + "lorem ipsum " * 2500
ONE_HOUR_MARK: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
FIVE_MINUTE_MARK: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "5m"}
DEFAULT_MARK: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
OVERLOAD_TOKEN: Final = "scripted-overload-on-this-turn"
BEDROCK_ONE_HOUR_MODEL: Final = "us.anthropic.claude-sonnet-5-5"
BEDROCK_FIVE_MINUTE_MODEL: Final = "anthropic.claude-sonnet-4-20250514-v1:0"
INFERENCE_PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/pinprofile"
_PIN_VALUE: Final = TypeAdapter(dict[str, str])
_JSON: Final = TypeAdapter(JsonValue)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class Pin:
    model_id: str
    ttl: int


def lives_one_hour(pin: Pin) -> bool:
    return FIVE_MINUTES < pin.ttl <= ONE_HOUR


def lives_five_minutes(pin: Pin) -> bool:
    return 0 < pin.ttl <= FIVE_MINUTES


def new_marker() -> str:
    return uuid.uuid4().hex


def marked(text: str, mark: JsonValue | None) -> dict[str, JsonValue]:
    block: Final[dict[str, JsonValue]] = {"type": "text", "text": text}
    return block if mark is None else {**block, "cache_control": mark}


def reference(session: str) -> str:
    return f"{LONG_TEXT} session {session}"


def question(marker: str) -> str:
    return f"Reply in one word. marker-{marker}"


def system_turn(session: str, mark: JsonValue | None) -> dict[str, JsonValue]:
    return {"role": "system", "content": [marked(reference(session), mark)]}


def ask_turn(marker: str, mark: JsonValue | None = None) -> dict[str, JsonValue]:
    if mark is None:
        return {"role": "user", "content": question(marker)}
    return {"role": "user", "content": [marked(question(marker), mark)]}


def follow_up(first_turn: Sequence[JsonValue], marker: str) -> list[JsonValue]:
    return [*first_turn, {"role": "assistant", "content": "sunny"}, ask_turn(marker)]


def chat_body(model: str, messages: Sequence[JsonValue], **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": list(messages), "max_tokens": 16, **fields}


def read_pins(redis: Redis) -> Mapping[str, Pin]:
    def pin_of(key: bytes) -> tuple[str, Pin] | None:
        value: Final = redis.get(key)
        ttl: Final = redis.ttl(key)
        if value is None or ttl < 0:
            return None
        return key.decode(), Pin(model_id=_PIN_VALUE.validate_json(value)["model_id"], ttl=ttl)

    found: Final = (pin_of(key) for key in redis.scan_iter(match="*:prompt_caching", count=1000))
    return MappingProxyType(dict(entry for entry in found if entry is not None))


def pins_since(redis: Redis, before: frozenset[str]) -> Mapping[str, Pin]:
    return MappingProxyType({key: pin for key, pin in read_pins(redis).items() if key not in before})


def one_new_pin(redis: Redis, before: frozenset[str]) -> tuple[str, Pin]:
    pins: Final = eventually(lambda: pins_since(redis, before), lambda found: len(found) >= 1, seconds=30)
    assert len(pins) == 1, pins
    return next(iter(pins.items()))


def pin_at(redis: Redis, key: str) -> Pin | None:
    return read_pins(redis).get(key)


def spend_rows(response_id: str) -> int:
    return len(read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)))


def spend_row_lands(response_id: str) -> None:
    eventually(lambda: spend_rows(response_id), lambda count: count >= 1, seconds=70)
    assert spend_rows(response_id) == 1


def end_user_row_lands(user: str) -> None:
    eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE end_user=%s', (user,)),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )
    assert len(read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE end_user=%s', (user,))) == 1


def no_pin_written(redis: Redis, before: frozenset[str], response_id: str) -> None:
    spend_row_lands(response_id)
    held(lambda: pins_since(redis, before), lambda found: not found, holding=3, seconds=20)


def _carries(value: JsonValue, ttl: str) -> bool:
    if isinstance(value, list):
        return any(_carries(item, ttl) for item in value)
    if not isinstance(value, dict):
        return False
    marks: Final = (value.get("cache_control"), value.get("cachePoint"))
    return any(isinstance(mark, dict) and mark.get("ttl") == ttl for mark in marks) or any(
        _carries(item, ttl) for item in value.values()
    )


def wire_carries_one_hour(request: Request) -> bool:
    return _carries(_JSON.validate_json(request.body), "1h")


def sent_with(wires: Sequence[Wire], marker: str) -> tuple[Request, ...]:
    return tuple(request for wire in wires for request in wire.drain() if marker_of(request) == marker)


def _overloaded() -> Reply:
    return Reply(
        status=529,
        body=json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}).encode(),
    )


def claude_peer(request: Request) -> Reply:
    if OVERLOAD_TOKEN.encode() in request.body:
        return _overloaded()
    return anthropic_peer(request)


def slow_claude_peer(request: Request) -> Reply:
    time.sleep(1)
    return claude_peer(request)


def vertex_peer(request: Request) -> Reply:
    if request.target == "/_oauth/token":
        return Reply(
            body=json.dumps(
                {"access_token": "synthetic-vertex-token", "token_type": "Bearer", "expires_in": 3600}
            ).encode()
        )
    return claude_peer(request)


def bedrock_runtime_peer(request: Request) -> Reply:
    if request.target.endswith("/converse"):
        return bedrock_peer(request)
    return claude_peer(request)


def openai_peer(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{marker_of(request)}",
                "object": "chat.completion",
                "created": 1760000000,
                "model": str(body.get("model")),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "sunny"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            }
        ).encode()
    )


def anthropic_deployment(name: str, identity: str, api_base: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": "anthropic/claude-opus-5-5",
            "api_base": api_base,
            "api_key": PROVIDER_KEY,
            **fields,
        },
        "model_info": {"id": identity},
    }


def bedrock_deployment(
    name: str, identity: str, model: str, api_base: str, model_info: Mapping[str, JsonValue] = MappingProxyType({})
) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": model,
            "api_key": BEDROCK_TOKEN,
            "aws_region_name": "us-east-1",
            "api_base": api_base,
            "aws_bedrock_runtime_endpoint": api_base,
        },
        "model_info": {"id": identity, **model_info},
    }


def deployment(name: str, identity: str, **litellm_params: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": dict(litellm_params), "model_info": {"id": identity}}


def vertex_deployment(name: str, identity: str, api_base: str) -> dict[str, JsonValue]:
    return deployment(
        name,
        identity,
        model="vertex_ai/claude-opus-5-5",
        api_base=api_base,
        vertex_project="scripted-project",
        vertex_location="us-east5",
        vertex_credentials=service_account_json("scripted-project", api_base),
    )


def pin_config(
    directory: Path,
    model_list: Sequence[JsonValue],
    *,
    litellm_settings: Mapping[str, JsonValue] = MappingProxyType({}),
    router_settings: Mapping[str, JsonValue] = MappingProxyType({}),
    general_settings: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    merged: Final = {
        **config,
        "model_list": list(model_list),
        "general_settings": {**object_value(config["general_settings"]), **general_settings},
        "litellm_settings": {**object_value(config["litellm_settings"]), **litellm_settings},
        "router_settings": {
            **object_value(config["router_settings"]),
            "num_retries": 0,
            "optional_pre_call_checks": ["prompt_caching"],
            **router_settings,
        },
    }
    path: Final = directory / f"prompt-cache-pins-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path


def post_chat(gateway: Gateway, body: Mapping[str, JsonValue]) -> tuple[str, str]:
    response: Final = gateway.request("POST", "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    return str(_JSON_OBJECT.validate_json(response.content)["id"]), response.headers["x-litellm-model-id"]
