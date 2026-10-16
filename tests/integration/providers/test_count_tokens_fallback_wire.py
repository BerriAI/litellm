import json
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final

import anthropic
import httpx
import pytest
from anthropic.types import MessageCountTokensToolParam, MessageTokensCount
from integration._support.bedrock_runtime_peer import target_of
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.ports import allocate_port
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_SONNET: Final = "us.anthropic.claude-sonnet-5-5"
_SONNET_BASE: Final = "anthropic.claude-sonnet-5-5"
_OPUS: Final = "us.anthropic.claude-opus-5-5"
_OPUS_BASE: Final = "anthropic.claude-opus-5-5"
_NOVA: Final = "amazon.nova-lite-v1:0"
_KIMI: Final = "us.moonshotai.kimi-k3"
_KIMI_BASE: Final = "moonshotai.kimi-k3"
_PROFILE_ID: Final = "lit9563count"
_PROFILE_ARN: Final = f"arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/{_PROFILE_ID}"
_COST_MAP_PATH: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
_REGION: Final = "us-east-1"
_ACCESS_KEY: Final = "AKIAINTEGRATIONCOUNT"
_SECRET_KEY: Final = "integration-count-secret"
_PROVIDER_COUNT: Final = 1345
_FALLBACK_COUNT: Final = 2718
_FRAMING: Final = 50
_UNSUPPORTED: Final = "The provided model doesn't support counting tokens."
_COUNT_TARGET: Final = re.compile(r"^/model/(.+)/count-tokens$")
_SCRIPTED_STATUS: Final = re.compile(r"status=(\d{3})")
_SCRIPTED_FALLBACK_STATUS: Final = re.compile(r"fallback=(\d{3})")
_CREDENTIAL: Final = re.compile(r"Credential=([^/]+)/")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, JsonValue]])

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
_OPENAI_PRIMARY: Final = "gpt-count-primary"
_OPENAI_FALLBACK: Final = "gpt-count-fallback"
_OPENAI_COUNT_TARGET: Final = "/v1/responses/input_tokens"
_COUNTED_FIELDS: Final = ("input", "instructions", "tools", "text")
_ANSWER_SCHEMA: Final[dict[str, JsonValue]] = {"type": "object", "properties": {"answer": {"type": "string"}}}
_ANTHROPIC_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Look up the current weather for a city",
    "input_schema": _WEATHER_SCHEMA,
}
_TOOL_TURNS: Final[list[JsonValue]] = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny"}]},
]
_LOCAL_SOURCE: Final = {"litellm_count_source": "local_estimate"}


def _prompt(status: int, fallback_status: int | None = None) -> str:
    unavailable: Final = "" if fallback_status is None else f" fallback={fallback_status}"
    return f"Count this message status={status}{unavailable}"


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


def _scripted(pattern: re.Pattern[str], request: Request, default: int) -> int:
    scripted: Final = pattern.search(request.body.decode(errors="replace"))
    return default if scripted is None else int(scripted.group(1))


def _runtime(request: Request) -> Reply:
    model: Final = _counted_model(request)
    if model == _OPUS_BASE:
        fallback_status: Final = _scripted(_SCRIPTED_FALLBACK_STATUS, request, 200)
        if fallback_status == 200:
            return _json_reply(200, {"inputTokens": _FALLBACK_COUNT})
        return _json_reply(fallback_status, {"message": f"scripted runtime {fallback_status} for {model}"})
    status: Final = _scripted(_SCRIPTED_STATUS, request, 500)
    if status == 200:
        return _json_reply(200, {"inputTokens": _PROVIDER_COUNT})
    return _json_reply(status, {"message": _UNSUPPORTED if status == 400 else f"scripted runtime {status}"})


def _mantle(request: Request) -> Reply:
    model: Final = _JSON_OBJECT.validate_json(request.body).get("model")
    return _json_reply(
        404, {"type": "error", "error": {"type": "not_found_error", "message": f"scripted mantle 404 for {model}"}}
    )


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
    return allocate_port()


@pytest.fixture(scope="module")
def mantle_port() -> int:
    return allocate_port()


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


def _count(gateway: Gateway, model: str, status: int, fallback_status: int | None = None) -> MessageTokensCount:
    with anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
        return client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": _prompt(status, fallback_status)}],
            system=_SYSTEM,
            tools=_SDK_TOOLS,
        )


def _messages_estimate(gateway: Gateway, model: str) -> tuple[int, JsonValue]:
    counted: Final = _count(gateway, model, 400)
    return counted.input_tokens, (counted.model_extra or {}).get("litellm_count_source")


def _utils_estimate(gateway: Gateway, model: str) -> tuple[int, JsonValue]:
    response: Final = gateway.request(
        "POST", "/utils/token_counter", {"model": model, "messages": [{"role": "user", "content": _prompt(400)}]}
    )
    assert response.status_code == 200, response.text
    body: Final = _JSON_OBJECT.validate_json(response.content)
    total: Final = body["total_tokens"]
    assert isinstance(total, int), body
    return total, body.get("count_source")


def _shipped_framing(model: str) -> JsonValue:
    return _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())[model].get("token_count_framing")


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


def test_fallback_model_404_answers_the_requested_models_error_and_skips_the_fallback_on_that_deployment(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        first: Final = _deployment(scenario, runtime_port)
        second: Final = _deployment(scenario, runtime_port)
        tried: Final = (_SONNET_BASE, _OPUS_BASE)
        skipped: Final = (_SONNET_BASE,)
        for model, expected in ((first, tried), (first, skipped), (second, tried)):
            with pytest.raises(anthropic.NotFoundError) as refused:
                _count(gateway, model, 404, fallback_status=404)
            assert _SONNET_BASE in str(refused.value) and "opus" not in str(refused.value).lower(), refused.value
            assert (_runtime_models(runtime), _mantle_models(mantle)) == (expected, expected)


@pytest.mark.parametrize("base_model_in", ["model_info", "litellm_params"])
def test_inference_profile_arn_counts_with_the_fallback_of_its_base_model(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int, base_model_in: str
) -> None:
    gateway: Final = counting_proxy.gateway
    base_model: Final[dict[str, JsonValue]] = {"base_model": f"bedrock/{_SONNET}"}
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model_info=base_model if base_model_in == "model_info" else None,
            api_key=None,
            **_litellm_params(_PROFILE_ARN, _url(runtime_port)),
            **(base_model if base_model_in == "litellm_params" else {}),
        )
        counted: Final = _count(gateway, model, 404)
        assert (counted.input_tokens, counted.model_extra) == (_FALLBACK_COUNT, _FALLBACK_SOURCE), counted
        assert _runtime_models(runtime) == (_PROFILE_ID, _OPUS_BASE)
        assert _mantle_models(mantle) == ()


@pytest.mark.parametrize(
    ("estimate", "runtime_calls"),
    [(_messages_estimate, (_KIMI_BASE, _KIMI_BASE)), (_utils_estimate, ())],
    ids=["messages_count_tokens", "utils_token_counter"],
)
def test_shipped_framing_is_added_to_the_estimate_and_a_configured_zero_turns_it_off(
    counting_proxy: OwnedProxy,
    runtime_port: int,
    mantle_port: int,
    estimate: Callable[[Gateway, str], tuple[int, JsonValue]],
    runtime_calls: tuple[str, ...],
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        shipped: Final = estimate(gateway, _deployment(scenario, runtime_port, _KIMI))
        unframed: Final = estimate(gateway, _deployment(scenario, runtime_port, _KIMI, {"token_count_framing": 0}))
        assert (shipped[1], unframed[1]) == ("local_estimate", "local_estimate"), (shipped, unframed)
        framing: Final = _shipped_framing(_KIMI)
        assert isinstance(framing, int) and framing > 0, framing
        assert shipped[0] - unframed[0] == framing, (shipped, unframed)
        assert _runtime_models(runtime) == runtime_calls
        assert _mantle_models(mantle) == ()


def _openai_count(request: Request) -> Reply:
    model: Final = _JSON_OBJECT.validate_json(request.body).get("model")
    if request.target == _OPENAI_COUNT_TARGET and model == _OPENAI_FALLBACK:
        return _json_reply(200, {"object": "response.input_tokens", "input_tokens": _FALLBACK_COUNT})
    return _json_reply(
        404,
        {"error": {"message": f"scripted openai 404 for {model}", "type": "invalid_request_error", "code": None}},
    )


def _counted_fields(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: body[key] for key in _COUNTED_FIELDS if key in body}


def _openai_counts(wire: Wire) -> tuple[tuple[str, JsonValue, dict[str, JsonValue]], ...]:
    bodies: Final = tuple((request.target, _JSON_OBJECT.validate_json(request.body)) for request in wire.drain())
    return tuple((target, body.get("model"), _counted_fields(body)) for target, body in bodies)


def _openai_deployment(scenario: Scenario, wire: Wire, model_info: Mapping[str, JsonValue]) -> str:
    return scenario.model(model=f"openai/{_OPENAI_PRIMARY}", api_base=f"{wire.url}/v1", model_info=model_info)


def test_fallback_model_counts_the_anthropic_body_converted_like_the_provider_count(
    counting_proxy: OwnedProxy,
) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_openai_count) as openai, gateway.scenario() as scenario:
        model: Final = _openai_deployment(scenario, openai, {"token_count_fallback_model": _OPENAI_FALLBACK})
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": model,
                "system": _SYSTEM,
                "tools": [_ANTHROPIC_TOOL],
                "output_config": {"format": {"type": "json_schema", "schema": _ANSWER_SCHEMA}},
                "messages": _TOOL_TURNS,
            },
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content) == {
            "input_tokens": _FALLBACK_COUNT,
            "litellm_count_source": "fallback_model",
            "litellm_count_fallback_model": _OPENAI_FALLBACK,
        }
        counts: Final = _openai_counts(openai)
        assert tuple((target, counted) for target, counted, _ in counts) == (
            (_OPENAI_COUNT_TARGET, _OPENAI_PRIMARY),
            (_OPENAI_COUNT_TARGET, _OPENAI_FALLBACK),
        ), counts
        primary: Final = counts[0][2]
        assert counts[1][2] == primary
        assert primary["instructions"] == _SYSTEM, primary
        text: Final = primary.get("text")
        assert isinstance(text, dict) and isinstance(text.get("format"), dict), primary
        input_items: Final = primary["input"]
        assert isinstance(input_items, list), primary
        item_types: Final = {item.get("type") for item in input_items if isinstance(item, dict)}
        assert {"function_call", "function_call_output"} <= item_types, input_items


@pytest.mark.parametrize(
    ("body", "named"),
    [
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Summarize the notes."},
                            {
                                "type": "search_result",
                                "source": "https://example.com/notes",
                                "title": "Notes",
                                "content": [],
                            },
                        ],
                    }
                ]
            },
            "search_result",
        ),
        (
            {"messages": [{"role": "user", "content": "hello"}, {"content": "no role"}]},
            "messages.1.role: Field required",
        ),
    ],
    ids=["unconvertible_block", "missing_role"],
)
def test_framing_keeps_the_400_for_a_body_the_openai_count_rejects_before_calling_openai(
    counting_proxy: OwnedProxy, body: Mapping[str, JsonValue], named: str
) -> None:
    gateway: Final = counting_proxy.gateway
    with wire_server(_openai_count) as openai, gateway.scenario() as scenario:
        model: Final = _openai_deployment(scenario, openai, {"token_count_framing": _FRAMING})
        rejected: Final = gateway.request("POST", "/v1/messages/count_tokens", {"model": model, **body})
        assert rejected.status_code == 400, rejected.text
        error: Final = _JSON_OBJECT.validate_json(rejected.content)["error"]
        assert isinstance(error, dict) and named in str(error.get("message")), error
        assert openai.drain() == ()
        estimated: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", {"model": model, "messages": [{"role": "user", "content": "hello"}]}
        )
        assert estimated.status_code == 200, estimated.text
        assert _JSON_OBJECT.validate_json(estimated.content).get("litellm_count_source") == "local_estimate"
        assert tuple(counted for _, counted, _ in _openai_counts(openai)) == (_OPENAI_PRIMARY,)


@pytest.mark.parametrize(
    ("fallback_status", "status", "error_type", "message"),
    [
        (404, 404, "not_found_error", f"scripted mantle 404 for {_SONNET_BASE}"),
        (429, 429, "rate_limit_error", f"scripted runtime 429 for {_OPUS_BASE}"),
    ],
    ids=["fallback_404_answers_the_requested_models_error", "fallback_429_answers_its_own_error"],
)
def test_fallback_failure_reaches_the_client_in_the_anthropic_error_envelope(
    counting_proxy: OwnedProxy,
    runtime_port: int,
    mantle_port: int,
    fallback_status: int,
    status: int,
    error_type: str,
    message: str,
) -> None:
    gateway: Final = counting_proxy.gateway
    with (
        wire_server(_runtime, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port),
        gateway.scenario() as scenario,
    ):
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": _deployment(scenario, runtime_port),
                "system": _SYSTEM,
                "tools": [_ANTHROPIC_TOOL],
                "messages": [{"role": "user", "content": _prompt(404, fallback_status)}],
            },
        )
        assert response.status_code == status, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        error: Final = body["error"]
        assert body["type"] == "error" and isinstance(error, dict), body
        assert (error.get("type"), error.get("message")) == (error_type, message), body
        assert _runtime_models(runtime) == (_SONNET_BASE, _OPUS_BASE)


def _runtime_cannot_count(_request: Request) -> Reply:
    return _json_reply(400, {"message": _UNSUPPORTED})


_REPLAYED_WEB_SEARCH: Final[list[JsonValue]] = [
    {"role": "user", "content": "What is the newest LiteLLM release?"},
    {
        "role": "assistant",
        "content": [
            {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "LiteLLM"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://docs.litellm.ai/release_notes",
                        "title": "Release notes",
                        "encrypted_content": "",
                    }
                ],
            },
            {"type": "text", "text": "The newest release is in the release notes."},
        ],
    },
    {"role": "user", "content": "Thanks."},
]
_BOOLEAN_SUBSCHEMA_TOOL: Final[dict[str, JsonValue]] = {
    "name": "lookup",
    "description": "Look up a value",
    "input_schema": {"type": "object", "properties": {"query": True}},
}
_UNREADABLE_BODIES: Final[dict[str, dict[str, JsonValue]]] = {
    "web_search_replay": {"messages": _REPLAYED_WEB_SEARCH},
    "boolean_subschema_tool": {
        "messages": [{"role": "user", "content": "Look up the value."}],
        "tools": [_BOOLEAN_SUBSCHEMA_TOOL],
    },
}
_COUNT_ENDPOINTS: Final[dict[str, tuple[str, str]]] = {
    "messages_count_tokens": ("/v1/messages/count_tokens", "litellm_count_source"),
    "utils_token_counter": ("/utils/token_counter?call_endpoint=true", "count_source"),
}


def _count_answer(response: httpx.Response, source_key: str) -> tuple[int, str]:
    if response.status_code not in (200, 400):
        return response.status_code, response.text
    body: Final = _JSON_OBJECT.validate_json(response.content)
    error: Final = body.get("error")
    if response.status_code == 400 and isinstance(error, dict):
        return 400, str(error.get("message"))
    return response.status_code, str(body.get(source_key))


@pytest.mark.parametrize("endpoint", sorted(_COUNT_ENDPOINTS))
@pytest.mark.parametrize("unreadable", sorted(_UNREADABLE_BODIES))
def test_shipped_framing_never_answers_a_500_for_a_body_the_local_count_cannot_read(
    counting_proxy: OwnedProxy, runtime_port: int, mantle_port: int, unreadable: str, endpoint: str
) -> None:
    gateway: Final = counting_proxy.gateway
    path, source_key = _COUNT_ENDPOINTS[endpoint]
    with (
        wire_server(_runtime_cannot_count, port=runtime_port) as runtime,
        wire_server(_mantle, port=mantle_port) as mantle,
        gateway.scenario() as scenario,
    ):
        response: Final = gateway.request(
            "POST", path, {"model": _deployment(scenario, runtime_port, _KIMI), **_UNREADABLE_BODIES[unreadable]}
        )
        status, answer = _count_answer(response, source_key)
        assert (status, answer) == (200, "local_estimate") or (status == 400 and _UNSUPPORTED in answer), (
            response.status_code,
            response.text,
        )
        assert _runtime_models(runtime) == (_KIMI_BASE,)
        assert _mantle_models(mantle) == ()


_GEMINI_MODEL: Final = "gemini-4-argon"
_VERTEX_PROJECT: Final = "scripted-project"
_UNCOUNTABLE_TOOL: Final[dict[str, JsonValue]] = {"name": "lookup", "input_schema": "not-a-schema"}


def _google_cannot_count(_request: Request) -> Reply:
    return _json_reply(400, {"error": {"code": 400, "message": "scripted count 400", "status": "INVALID_ARGUMENT"}})


def _google_deployment(gateway: Gateway, scenario: Scenario, wire: Wire, provider: str) -> str:
    framing: Final[dict[str, JsonValue]] = {"token_count_framing": _FRAMING}
    if provider == "gemini":
        return scenario.model(
            model=f"gemini/{_GEMINI_MODEL}", api_base=wire.url, api_key="scripted-gemini-key", model_info=framing
        )
    return scenario.model(
        model=f"vertex_ai/{_GEMINI_MODEL}",
        api_base=wire.url,
        api_key=None,
        vertex_project=_VERTEX_PROJECT,
        vertex_location="us-east5",
        vertex_credentials=service_account_json(_VERTEX_PROJECT, gateway.upstream_url.rstrip("/")),
        model_info=framing,
    )


def _google_count_targets(wire: Wire) -> tuple[bool, ...]:
    return tuple(request.target.endswith(":countTokens") for request in wire.drain())


@pytest.mark.parametrize("provider", ["gemini", "vertex_ai"])
def test_framing_keeps_the_400_for_tools_the_gemini_count_rejects_before_calling_google(
    counting_proxy: OwnedProxy, provider: str
) -> None:
    gateway: Final = counting_proxy.gateway
    hello: Final[list[JsonValue]] = [{"role": "user", "content": "hello"}]
    with wire_server(_google_cannot_count) as google, gateway.scenario() as scenario:
        model: Final = _google_deployment(gateway, scenario, google, provider)
        rejected: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", {"model": model, "messages": hello, "tools": [_UNCOUNTABLE_TOOL]}
        )
        assert rejected.status_code == 400, rejected.text
        error: Final = _JSON_OBJECT.validate_json(rejected.content)["error"]
        assert isinstance(error, dict) and "JSON object schema" in str(error.get("message")), error
        assert _google_count_targets(google) == ()
        estimated: Final = gateway.request("POST", "/v1/messages/count_tokens", {"model": model, "messages": hello})
        assert estimated.status_code == 200, estimated.text
        assert _JSON_OBJECT.validate_json(estimated.content).get("litellm_count_source") == "local_estimate"
        assert _google_count_targets(google) == (True,)
