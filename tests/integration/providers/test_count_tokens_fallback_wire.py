import json
import re
from collections.abc import Iterator, Mapping
from typing import Final

import anthropic
import pytest
from anthropic.types import MessageCountTokensToolParam, MessageTokensCount
from integration._support.bedrock_runtime_peer import target_of
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import free_loopback_port
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_SONNET: Final = "us.anthropic.claude-sonnet-5-5"
_SONNET_BASE: Final = "anthropic.claude-sonnet-5-5"
_OPUS: Final = "us.anthropic.claude-opus-5-5"
_OPUS_BASE: Final = "anthropic.claude-opus-5-5"
_NOVA: Final = "amazon.nova-lite-v1:0"
_REGION: Final = "us-east-1"
_ACCESS_KEY: Final = "AKIAINTEGRATIONCOUNT"
_SECRET_KEY: Final = "integration-count-secret"
_PROVIDER_COUNT: Final = 1345
_FALLBACK_COUNT: Final = 2718
_FRAMING: Final = 50
_UNSUPPORTED: Final = "The provided model doesn't support counting tokens."
_COUNT_TARGET: Final = re.compile(r"^/model/(.+)/count-tokens$")
_SCRIPTED_STATUS: Final = re.compile(r"status=(\d{3})")
_CREDENTIAL: Final = re.compile(r"Credential=([^/]+)/")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

_SYSTEM: Final = "You are a terse assistant that answers in one sentence"
_WEATHER_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City to look up"}},
    "required": ["city"],
}
_SDK_TOOLS: Final[list[MessageCountTokensToolParam]] = [
    {
        "name": "get_weather",
        "description": "Look up the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City to look up"}},
            "required": ["city"],
        },
    }
]
_FALLBACK_SOURCE: Final = {"litellm_count_source": "fallback_model", "litellm_count_fallback_model": _OPUS}
_LOCAL_SOURCE: Final = {"litellm_count_source": "local_estimate"}


def _prompt(status: int) -> str:
    return f"Count this message status={status}"


def _converse_body(prompt: str) -> dict[str, JsonValue]:
    return {
        "input": {
            "converse": {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "system": [{"text": _SYSTEM}],
                "toolConfig": {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "get_weather",
                                "description": "Look up the current weather for a city",
                                "inputSchema": {"json": _WEATHER_SCHEMA},
                            }
                        }
                    ]
                },
            }
        }
    }


def _json_reply(status: int, payload: Mapping[str, JsonValue]) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def _counted_model(request: Request) -> str:
    counted: Final = _COUNT_TARGET.match(target_of(request))
    return target_of(request) if counted is None else counted.group(1)


def _runtime(request: Request) -> Reply:
    if _counted_model(request) == _OPUS_BASE:
        return _json_reply(200, {"inputTokens": _FALLBACK_COUNT})
    scripted: Final = _SCRIPTED_STATUS.search(request.body.decode(errors="replace"))
    status: Final = 500 if scripted is None else int(scripted.group(1))
    if status == 200:
        return _json_reply(200, {"inputTokens": _PROVIDER_COUNT})
    return _json_reply(status, {"message": _UNSUPPORTED if status == 400 else f"scripted runtime {status}"})


def _mantle(_request: Request) -> Reply:
    return _json_reply(404, {"type": "error", "error": {"type": "not_found_error", "message": "scripted mantle 404"}})


def _signer(request: Request) -> str:
    credential: Final = _CREDENTIAL.search(request.headers.get("authorization", ""))
    return "" if credential is None else credential.group(1)


def _runtime_counts(wire: Wire) -> tuple[tuple[str, str, dict[str, JsonValue]], ...]:
    return tuple(
        (_counted_model(request), _signer(request), _JSON_OBJECT.validate_json(request.body))
        for request in wire.drain()
    )


def _runtime_models(wire: Wire) -> tuple[str, ...]:
    return tuple(_counted_model(request) for request in wire.drain())


def _mantle_models(wire: Wire) -> tuple[JsonValue, ...]:
    return tuple(_JSON_OBJECT.validate_json(request.body).get("model") for request in wire.drain())


def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _litellm_params(model: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "model": f"bedrock/{model}",
        "api_base": api_base,
        "aws_access_key_id": _ACCESS_KEY,
        "aws_secret_access_key": _SECRET_KEY,
        "aws_region_name": _REGION,
    }


@pytest.fixture(scope="module")
def runtime_port() -> int:
    return free_loopback_port()


@pytest.fixture(scope="module")
def mantle_port() -> int:
    return free_loopback_port()


@pytest.fixture(scope="module")
def counting_proxy(tmp_path_factory: pytest.TempPathFactory, mantle_port: int) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            tmp_path_factory.mktemp("count-fallback"),
            {"BEDROCK_MANTLE_API_BASE": _url(mantle_port), "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
            workers=1,
            remove_environment=("AWS_BEARER_TOKEN_BEDROCK",),
        ) as owned,
    ):
        yield owned


def _deployment(
    scenario: Scenario, runtime_port: int, model: str = _SONNET, model_info: Mapping[str, JsonValue] | None = None
) -> str:
    return scenario.model(model_info=model_info, api_key=None, **_litellm_params(model, _url(runtime_port)))


def _count(gateway: Gateway, model: str, status: int) -> MessageTokensCount:
    with anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
        return client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": _prompt(status)}], system=_SYSTEM, tools=_SDK_TOOLS
        )


def test_model_neither_bedrock_path_counts_returns_the_cost_map_fallback_count(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        counted: Final = _count(gateway, _deployment(scenario, runtime_port), 404)
        assert (counted.input_tokens, counted.model_extra) == (_FALLBACK_COUNT, _FALLBACK_SOURCE), counted
        assert _runtime_counts(runtime) == (
            (_SONNET_BASE, _ACCESS_KEY, _converse_body(_prompt(404))),
            (_OPUS_BASE, _ACCESS_KEY, _converse_body(_prompt(404))),
        )
        assert _mantle_models(mantle) == (_SONNET_BASE,)


def test_runtime_401_raises_without_counting_the_fallback_model(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = _deployment(scenario, runtime_port)
        assert _count(gateway, model, 404).model_extra == _FALLBACK_SOURCE
        assert _runtime_models(runtime) == (_SONNET_BASE, _OPUS_BASE)
        assert _mantle_models(mantle) == (_SONNET_BASE,)
        with pytest.raises(anthropic.AuthenticationError):
            _count(gateway, model, 401)
        assert _runtime_models(runtime) == (_SONNET_BASE,)
        assert _mantle_models(mantle) == ()


def test_framing_is_added_to_the_local_estimate_when_the_runtime_cannot_count(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        framed: Final = _count(
            gateway, _deployment(scenario, runtime_port, _NOVA, {"token_count_framing": _FRAMING}), 400
        )
        unframed: Final = _count(gateway, _deployment(scenario, runtime_port, _NOVA, {"token_count_framing": 0}), 400)
        assert (framed.model_extra, unframed.model_extra) == (_LOCAL_SOURCE, _LOCAL_SOURCE), (framed, unframed)
        assert framed.input_tokens - unframed.input_tokens == _FRAMING, (framed, unframed)
        assert _runtime_models(runtime) == (_NOVA, _NOVA)
        assert _mantle_models(mantle) == ()


def test_framing_on_one_deployment_leaves_a_sibling_of_the_same_model_raising_the_400(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        framed: Final = _deployment(scenario, runtime_port, _NOVA, {"token_count_framing": _FRAMING})
        sibling: Final = _deployment(scenario, runtime_port, _NOVA)
        assert _count(gateway, framed, 400).model_extra == _LOCAL_SOURCE
        with pytest.raises(anthropic.BadRequestError, match=re.escape(_UNSUPPORTED)):
            _count(gateway, sibling, 400)
        assert _runtime_models(runtime) == (_NOVA, _NOVA)
        assert _mantle_models(mantle) == ()


def test_provider_count_is_tagged_provider_without_a_fallback_model(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        counted: Final = _count(gateway, _deployment(scenario, runtime_port), 200)
        assert (counted.input_tokens, counted.model_extra) == (_PROVIDER_COUNT, {"litellm_count_source": "provider"})
        assert _runtime_counts(runtime) == ((_SONNET_BASE, _ACCESS_KEY, _converse_body(_prompt(200))),)
        assert _mantle_models(mantle) == ()
