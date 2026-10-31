import itertools
import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, held
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.prometheus_series import Sample, scrape, series_cap_config
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.worker_readiness import wait_for_worker_readiness
from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

Endpoint: TypeAlias = Literal["chat", "responses", "messages"]
Deploy: TypeAlias = Callable[[Gateway, Scenario, Wire], str]

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROJECT: Final = "scripted-project"
_REPLY_TEXT: Final = "header probe reply"
_USAGE: Final[JsonValue] = {"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17}
_UPSTREAM_COST: Final = "0.0421"
_SPEND_SELECT: Final = 'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s'
_KEY_SPEND_SELECT: Final = 'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s'
_PATHS: Final[Mapping[Endpoint, str]] = MappingProxyType(
    {"chat": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}
)
_RATE_LIMIT_HEADERS: Final[Mapping[str, str]] = MappingProxyType(
    {"x-ratelimit-remaining-requests": "12.5", "x-ratelimit-remaining-tokens": "856, abc"}
)
_METERED_WORKERS: Final = 2
_METERED_CALLS: Final[tuple[tuple[Endpoint, str, bool], ...]] = (
    ("chat", "groq", False),
    ("chat", "groq", True),
    ("chat", "openai", False),
    ("chat", "openai", True),
    ("responses", "groq", False),
    ("responses", "groq", True),
    ("messages", "groq", False),
    ("messages", "groq", True),
)
_LATIN1_CALLS: Final[tuple[tuple[Endpoint, str, bool], ...]] = (
    ("chat", "bedrock_mantle", False),
    ("chat", "bedrock_mantle", True),
    ("responses", "groq", False),
    ("responses", "groq", True),
    ("messages", "groq", False),
    ("messages", "groq", True),
)


def _keyed(model: str, **extra: JsonValue) -> Deploy:
    def create(_gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
        return scenario.model(model=model, api_base=wire.url, api_key="synthetic-provider-key", **extra)

    return create


def _vertex(model: str, location: str) -> Deploy:
    def create(gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
        return scenario.model(
            model=f"vertex_ai/{model}",
            api_base=wire.url,
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=location,
            vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        )

    return create


_MEMBERS: Final[Mapping[str, Deploy]] = MappingProxyType(
    {
        "bedrock_mantle": _keyed("bedrock_mantle/google.gemma-4-31b", aws_region_name="us-west-2"),
        "groq": _keyed("groq/openai/gpt-oss-20b"),
        "openai": _keyed("openai/gpt-5.4-mini"),
        "watsonx": _keyed("watsonx/ibm/granite-3-8b-instruct", zen_api_key="synthetic-zen-key", project_id=_PROJECT),
        "vertex_model_garden": _vertex("xai/grok-4.20-non-reasoning", "global"),
        "maritalk": _keyed("maritalk/sabia-3"),
    }
)


def _reply_json() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "scripted",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": "stop"}
            ],
            "usage": _USAGE,
            "service_tier": "on_demand",
        }
    ).encode()


def _chunk(response_id: str, choices: list[JsonValue], **extra: JsonValue) -> bytes:
    body: Final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "scripted",
        "choices": choices,
        **extra,
    }
    return f"data: {json.dumps(body)}\n\n".encode()


def _frames() -> tuple[bytes, ...]:
    response_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    return (
        _chunk(
            response_id, [{"index": 0, "delta": {"role": "assistant", "content": _REPLY_TEXT}, "finish_reason": None}]
        ),
        _chunk(response_id, [{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        _chunk(response_id, [], usage=_USAGE),
        b"data: [DONE]\n\n",
    )


def _peer(headers: Mapping[str, str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
            return Reply(content_type="text/event-stream", chunks=_frames(), headers=headers)
        return Reply(body=_reply_json(), headers=headers)

    return respond


def _request_id_headers(request_id: str) -> Mapping[str, str]:
    return MappingProxyType({"x-request-id": request_id, "x-amzn-requestid": request_id})


def _body(endpoint: Endpoint, model: str, stream: bool) -> dict[str, JsonValue]:
    prompt: Final = f"header probe {uuid.uuid4().hex}"
    match endpoint:
        case "chat":
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "cache": {"no-cache": True},
                **({"stream_options": {"include_usage": True}} if stream else {}),
            }
        case "responses":
            return {"model": model, "input": prompt, "stream": stream}
        case "messages":
            return {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
            }


def _send(gateway: Gateway, endpoint: Endpoint, model: str, key: str, *, stream: bool = False) -> httpx.Response:
    return gateway.request("POST", _PATHS[endpoint], _body(endpoint, model, stream), key=key)


class _Reply(BaseModel):
    model_config = ConfigDict(frozen=True)


class _Text(_Reply):
    text: str = ""


class _ChatMessage(_Reply):
    content: str = ""


class _ChatChoice(_Reply):
    message: _ChatMessage


class _ChatReply(_Reply):
    choices: tuple[_ChatChoice, ...]


class _ResponsesItem(_Reply):
    content: tuple[_Text, ...] = ()


class _ResponsesReply(_Reply):
    output: tuple[_ResponsesItem, ...]


class _MessagesReply(_Reply):
    content: tuple[_Text, ...]


def _text(endpoint: Endpoint, response: httpx.Response) -> str:
    match endpoint:
        case "chat":
            return _ChatReply.model_validate_json(response.content).choices[0].message.content
        case "responses":
            items: Final = _ResponsesReply.model_validate_json(response.content).output
            return "".join(part.text for part in itertools.chain.from_iterable(item.content for item in items))
        case "messages":
            return "".join(block.text for block in _MessagesReply.model_validate_json(response.content).content)


def _header_cost(response: httpx.Response) -> float:
    values: Final = response.headers.get_list("x-litellm-response-cost")
    return float(values[0]) if values else math.nan


def _llm_provider_request_ids(response: httpx.Response) -> dict[str, str | None]:
    return {name: response.headers.get(name) for name in ("llm_provider-x-request-id", "llm_provider-x-amzn-requestid")}


def _spend(key: str) -> float:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda rows: len(rows) >= 1, seconds=70)
    rows: Final = held(lambda: read_rows(_SPEND_SELECT, (digest,)), lambda found: len(found) == 1, holding=3)
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float)), rows
    return float(spend)


def _model_values(samples: Sequence[Sample], name: str, model: str) -> tuple[float, ...]:
    return tuple(
        sample.value for sample in samples if sample.name == name and sample.labels.get("model_group") == model
    )


def _deployment_metrics(
    samples: Sequence[Sample], models: Mapping[str, str]
) -> dict[str, tuple[float, tuple[float, ...], tuple[float, ...]]]:
    return {
        call: (
            sum(_model_values(samples, "litellm_deployment_success_responses_total", model)),
            _model_values(samples, "litellm_remaining_requests_metric", model),
            _model_values(samples, "litellm_remaining_tokens_metric", model),
        )
        for call, model in models.items()
    }


def _key_spend(key: str) -> float | None:
    rows: Final = read_rows(_KEY_SPEND_SELECT, (sha256(key.encode()).hexdigest(),))
    spend: Final = rows[0]["spend"] if rows else None
    return float(spend) if isinstance(spend, (int, float)) else None


@pytest.mark.parametrize(
    ("endpoint", "member"),
    [
        ("chat", "bedrock_mantle"),
        ("chat", "groq"),
        ("chat", "watsonx"),
        ("chat", "vertex_model_garden"),
        ("chat", "maritalk"),
        ("responses", "groq"),
    ],
)
def test_non_stream_returns_the_provider_request_id(gateway: Gateway, endpoint: Endpoint, member: str) -> None:
    request_id: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(_peer(_request_id_headers(request_id))) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        response: Final = _send(gateway, endpoint, model, scenario.key(models=[model]))
        assert response.status_code == 200, response.text
        assert _text(endpoint, response) == _REPLY_TEXT, response.text
        assert _llm_provider_request_ids(response) == {
            "llm_provider-x-request-id": request_id,
            "llm_provider-x-amzn-requestid": request_id,
        }, dict(response.headers)
        assert len(wire.drain()) == 1


def test_stream_chat_returns_the_provider_request_id(gateway: Gateway) -> None:
    request_id: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(_peer(_request_id_headers(request_id))) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["bedrock_mantle"](gateway, scenario, wire)
        response: Final = _send(gateway, "chat", model, scenario.key(models=[model]), stream=True)
        assert response.status_code == 200, response.text
        assert _REPLY_TEXT in response.text, response.text
        assert _llm_provider_request_ids(response) == {
            "llm_provider-x-request-id": request_id,
            "llm_provider-x-amzn-requestid": request_id,
        }, dict(response.headers)
        assert len(wire.drain()) == 1


def test_upstream_cannot_overwrite_litellm_owned_headers(gateway: Gateway) -> None:
    spoofed: Final = {
        "x-litellm-call-id": "spoofed-call-id",
        "x-litellm-model-id": "spoofed-model-id",
        "x-litellm-attempted-fallbacks": "7",
    }
    with wire_server(_peer(spoofed)) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS["groq"](gateway, scenario, wire)
        response: Final = _send(gateway, "chat", model, scenario.key(models=[model]))
        assert response.status_code == 200, response.text
        assert response.headers["x-litellm-call-id"] != spoofed["x-litellm-call-id"], dict(response.headers)
        assert response.headers["x-litellm-model-id"] != spoofed["x-litellm-model-id"], dict(response.headers)
        assert response.headers["x-litellm-attempted-fallbacks"] == "0", dict(response.headers)
        assert len(wire.drain()) == 1


@pytest.mark.parametrize(("endpoint", "member"), [("chat", "bedrock_mantle"), ("responses", "groq")])
def test_non_stream_spend_matches_stream_spend_under_an_upstream_cost_header(
    gateway: Gateway, endpoint: Endpoint, member: str
) -> None:
    with wire_server(_peer({"x-litellm-response-cost": _UPSTREAM_COST})) as wire, gateway.scenario() as scenario:
        model: Final = _MEMBERS[member](gateway, scenario, wire)
        stream_key: Final = scenario.key(models=[model])
        non_stream_key: Final = scenario.key(models=[model])
        streamed: Final = _send(gateway, endpoint, model, stream_key, stream=True)
        assert streamed.status_code == 200, streamed.text
        assert _REPLY_TEXT in streamed.text, streamed.text
        whole: Final = _send(gateway, endpoint, model, non_stream_key)
        assert whole.status_code == 200, whole.text
        assert _text(endpoint, whole) == _REPLY_TEXT, whole.text
        stream_spend: Final = _spend(stream_key)
        assert stream_spend > 0
        non_stream_spend: Final = _spend(non_stream_key)
        assert math.isclose(non_stream_spend, stream_spend), (non_stream_spend, stream_spend)
        assert len(wire.drain()) == 2


@pytest.mark.parametrize("member", ["openai", "groq"])
@pytest.mark.parametrize("upstream_cost", ["-5", "abc", "nan", "inf"], ids=["negative", "garbage", "nan", "infinite"])
def test_an_invalid_upstream_cost_header_bills_the_price_map(gateway: Gateway, member: str, upstream_cost: str) -> None:
    with (
        wire_server(_peer({"x-litellm-response-cost": upstream_cost})) as invalid_wire,
        wire_server(_peer({})) as plain_wire,
        gateway.scenario() as scenario,
    ):
        invalid_model: Final = _MEMBERS[member](gateway, scenario, invalid_wire)
        plain_model: Final = _MEMBERS[member](gateway, scenario, plain_wire)
        budget_key: Final = scenario.key(models=[invalid_model], max_budget=1)
        reference_key: Final = scenario.key(models=[plain_model])
        billed: Final = _send(gateway, "chat", invalid_model, budget_key)
        reference: Final = _send(gateway, "chat", plain_model, reference_key)
        assert (billed.status_code, reference.status_code) == (200, 200), (billed.text, reference.text)
        reference_spend: Final = _spend(reference_key)
        assert reference_spend > 0
        key_spend: Final = eventually(
            lambda: _key_spend(budget_key),
            lambda spend: spend is not None and math.isclose(spend, reference_spend),
            70,
            return_last_on_timeout=True,
        )
        billed_costs: Final = {
            "response_header": _header_cost(billed),
            "spend_log": _spend(budget_key),
            "key": key_spend if key_spend is not None else math.nan,
        }
        assert {name: math.isclose(cost, reference_spend) for name, cost in billed_costs.items()} == dict.fromkeys(
            billed_costs, True
        ), (billed_costs, reference_spend)
        followup: Final = _send(gateway, "chat", invalid_model, budget_key)
        assert followup.status_code == 200, followup.text
        assert (len(invalid_wire.drain()), len(plain_wire.drain())) == (2, 1)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_provider_header_outside_latin1_does_not_fail_the_call(gateway: Gateway, tmp_path: Path) -> None:
    request_id: Final = f"req_{uuid.uuid4().hex}"
    utf8_bytes_as_wire_text: Final = "東京リージョン".encode().decode("latin-1")
    headers: Final = {**_request_id_headers(request_id), "x-provider-region": utf8_bytes_as_wire_text}
    with (
        wire_server(_peer(headers)) as wire,
        owned_proxy(gateway, tmp_path, {"DISABLE_AIOHTTP_TRANSPORT": "True"}) as httpx_transport,
        httpx_transport.scenario() as scenario,
    ):
        models: Final = {
            member: _MEMBERS[member](httpx_transport, scenario, wire) for member in ("bedrock_mantle", "groq")
        }
        key: Final = scenario.key(models=list(models.values()))
        replies: Final = {
            f"{endpoint}-{'stream' if stream else 'whole'}": _send(
                httpx_transport, endpoint, models[member], key, stream=stream
            )
            for endpoint, member, stream in _LATIN1_CALLS
        }
        assert {call: reply.status_code for call, reply in replies.items()} == dict.fromkeys(replies, 200), {
            call: reply.text for call, reply in replies.items()
        }
        assert {call: reply.headers.get("llm_provider-x-provider-region") for call, reply in replies.items()} == (
            dict.fromkeys(replies, None)
        )
        assert {call: _REPLY_TEXT in reply.text for call, reply in replies.items()} == dict.fromkeys(replies, True)
        assert {endpoint: _text(endpoint, replies[f"{endpoint}-whole"]) for endpoint in _PATHS} == dict.fromkeys(
            _PATHS, _REPLY_TEXT
        )
        forwarding: Final = ("chat-whole", "chat-stream", "responses-whole")
        assert {call: _llm_provider_request_ids(replies[call]) for call in forwarding} == dict.fromkeys(
            forwarding, {"llm_provider-x-request-id": request_id, "llm_provider-x-amzn-requestid": request_id}
        )
        assert len(wire.drain()) == len(_LATIN1_CALLS)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + 120)
def test_rate_limit_headers_set_the_remaining_gauges_and_keep_the_deployment_metrics(
    gateway: Gateway, tmp_path: Path
) -> None:
    multiproc_dir: Final = tmp_path / "prometheus"
    multiproc_dir.mkdir()
    with (
        wire_server(_peer(_RATE_LIMIT_HEADERS)) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {"PROMETHEUS_MULTIPROC_DIR": str(multiproc_dir)},
            config=series_cap_config(tmp_path, {}),
            workers=_METERED_WORKERS,
        ) as metered,
        metered.scenario() as scenario,
    ):
        calls: Final = {
            f"{endpoint}-{member}-{'stream' if stream else 'whole'}": (
                endpoint,
                _MEMBERS[member](metered, scenario, wire),
                stream,
            )
            for endpoint, member, stream in _METERED_CALLS
        }
        models: Final = {call: model for call, (_, model, _) in calls.items()}
        for model in models.values():
            wait_for_worker_readiness(metered, model, 1, _METERED_WORKERS)
        key: Final = scenario.key(models=list(models.values()))
        statuses: Final = {
            call: _send(metered, endpoint, model, key, stream=stream).status_code
            for call, (endpoint, model, stream) in calls.items()
        }
        assert statuses == dict.fromkeys(calls, 200)
        observed: Final = eventually(
            lambda: _deployment_metrics(scrape(metered), models),
            lambda metrics: all(successes >= 1 for successes, _, _ in metrics.values()),
            30,
            return_last_on_timeout=True,
        )
        assert observed == dict.fromkeys(calls, (1.0, (12.5,), ())), observed
        assert len(wire.drain()) == len(calls)
