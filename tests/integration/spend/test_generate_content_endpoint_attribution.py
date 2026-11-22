import json
import shutil
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import reduce
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, held
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, Field, JsonValue

_GENERATE_LABEL: Final = "/models/{model}:generateContent"
_STREAM_LABEL: Final = "/models/{model}:streamGenerateContent"
_CHAT_LABEL: Final = "/chat/completions"
_PREFIXES: Final = ("/v1beta/models", "/models")
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_SECONDS: Final = 70
_INTEGRATION_ROOT: Final = Path(__file__).resolve().parents[1]
_BASE_CONFIG: Final = _INTEGRATION_ROOT / "proxy_config.yaml"
_PROVIDER_MODULE: Final = _INTEGRATION_ROOT / "_support" / "custom_generate_content_provider.py"
_GEMINI_MODEL: Final = "gemini-3.8-flash"
_CHAT_USAGE: Final = {"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17}
_GEMINI_USAGE: Final = {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}
_GEMINI_REPLY: Final = {
    "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": _GEMINI_USAGE,
}
_GEMINI_FRAMES: Final = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"index":0}]}\n\n',
    b"data: "
    + json.dumps(
        {
            "candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": _GEMINI_USAGE,
        }
    ).encode()
    + b"\n\n",
)
_UPSTREAM_ERROR: Final = json.dumps(
    {"error": {"message": "scripted failure", "type": "invalid_request_error", "code": 400, "status": "FAILED"}}
).encode()
_FORGED_ROUTE_METADATA: Final = {
    "user_api_key_request_route": "/v1beta/models/x:generateContent",
    "route_call_type": "agenerate_content",
}
_ACCOUNTING_SETTINGS: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    "ledger_off": {"use_spend_logs_ledger": False},
    "metadata_status_only": {"spend_logs_metadata_fields": {"include": ["status"]}},
}
_SPEND_ROWS: Final = (
    'SELECT call_type, status, spend, prompt_tokens, completion_tokens, cache_hit FROM "LiteLLM_SpendLogs" '
    "WHERE api_key=%s"
)
_DAILY_ROWS: Final = 'SELECT endpoint FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s'


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _Metrics(_Frozen):
    spend: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class _Endpoint(_Frozen):
    metrics: _Metrics


class _Breakdown(_Frozen):
    endpoints: dict[str, _Endpoint] = Field(default_factory=dict)


class _Day(_Frozen):
    metrics: _Metrics
    breakdown: _Breakdown


class _Report(_Frozen):
    results: tuple[_Day, ...]


@dataclass(frozen=True, slots=True)
class _TeamActivity:
    totals: _Metrics
    endpoints: Mapping[str, _Metrics]


@dataclass(frozen=True, slots=True)
class _Backend:
    model: str
    api_root: str
    prompt_tokens: int
    completion_tokens: int
    chat: bool

    def target(self, stream: bool) -> str:
        return "/v1/chat/completions" if self.chat else f"/models/{_GEMINI_MODEL}:{_method(stream)}"

    def usage(self) -> _Metrics:
        return _Metrics(
            spend=self.prompt_tokens * _INPUT_COST + self.completion_tokens * _OUTPUT_COST,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            api_requests=1,
            successful_requests=1,
        )


@dataclass(frozen=True, slots=True)
class _Tenant:
    team: str
    key: str


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    wire: Wire


_NON_GEMINI: Final = _Backend("openai/gpt-5.4-mini", "/v1", 13, 4, True)
_GEMINI: Final = _Backend(f"gemini/{_GEMINI_MODEL}", "", 11, 7, False)
_BACKENDS: Final = pytest.mark.parametrize("backend", (_NON_GEMINI, _GEMINI), ids=("non_gemini", "gemini"))
_STREAMS: Final = pytest.mark.parametrize("stream", (False, True), ids=("generate", "stream"))
_EACH_PREFIX: Final = pytest.mark.parametrize("prefix", _PREFIXES, ids=("v1beta", "unprefixed"))


def _method(stream: bool) -> str:
    return "streamGenerateContent" if stream else "generateContent"


def _label(stream: bool) -> str:
    return _STREAM_LABEL if stream else _GENERATE_LABEL


def _chat_reply() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-5.4-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": _CHAT_USAGE,
        }
    ).encode()


def _chat_frames() -> tuple[bytes, ...]:
    common: Final = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-5.4-mini",
    }
    chunks: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": _CHAT_USAGE},
    )
    return (*(f"data: {json.dumps({**common, **chunk})}\n\n".encode() for chunk in chunks), b"data: [DONE]\n\n")


def _sse(chunks: tuple[bytes, ...], fault: str) -> Reply:
    return Reply(content_type="text/event-stream", chunks=chunks, abort_after=1 if fault == "cut" else None)


def _upstream(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    fault: Final = path.split("/")[1]
    if fault == "rejected":
        return Reply(status=400, body=_UPSTREAM_ERROR)
    if fault == "down":
        return Reply(status=500, body=_UPSTREAM_ERROR)
    if path.endswith("/complete"):
        return Reply(body=_chat_reply())
    if path.endswith(":streamGenerateContent"):
        return _sse(_GEMINI_FRAMES, fault)
    if path.endswith((":generateContent", "/generate")):
        return Reply(body=json.dumps(_GEMINI_REPLY).encode())
    if json.loads(request.body).get("stream") is True:
        return _sse(_chat_frames(), fault)
    return Reply(body=_chat_reply())


def _deployment(scenario: Scenario, wire: Wire, backend: _Backend, *, fault: str = "", name: str | None = None) -> str:
    return scenario.model(
        name=name,
        model=backend.model,
        api_base=f"{wire.url}{fault}{backend.api_root}",
        api_key="synthetic-provider-key",
        input_cost_per_token=_INPUT_COST,
        output_cost_per_token=_OUTPUT_COST,
        num_retries=0,
        max_retries=0,
    )


def _tenant(scenario: Scenario, *models: str) -> _Tenant:
    team: Final = scenario.team(models=list(models))
    return _Tenant(team, scenario.key(team_id=team, models=list(models)))


def _contents(text: str) -> list[JsonValue]:
    return [{"role": "user", "parts": [{"text": text}]}]


def _generate(
    gateway: Gateway,
    model: str,
    key: str,
    stream: bool,
    *,
    prefix: str = _PREFIXES[0],
    text: str | None = None,
    **fields: JsonValue,
) -> httpx.Response:
    query: Final = "?alt=sse" if stream else ""
    prompt: Final = text if text is not None else f"attribution probe {uuid.uuid4().hex}"
    return gateway.request(
        "POST", f"{prefix}/{model}:{_method(stream)}{query}", {"contents": _contents(prompt), **fields}, key=key
    )


def _chat(gateway: Gateway, model: str, key: str, stream: bool, **fields: JsonValue) -> httpx.Response:
    streaming: Final[dict[str, JsonValue]] = (
        {"stream": True, "stream_options": {"include_usage": True}} if stream else {}
    )
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"attribution probe {uuid.uuid4().hex}"}],
            **streaming,
            **fields,
        },
        key=key,
    )


def _plus(left: _Metrics, right: _Metrics) -> _Metrics:
    return _Metrics(
        **{name: getattr(left, name) + getattr(right, name) for name in _Metrics.model_fields},
    )


def _total(metrics: Iterable[_Metrics]) -> _Metrics:
    return reduce(_plus, metrics, _Metrics())


def _activity(gateway: Gateway, team: str) -> _TeamActivity:
    today: Final = datetime.now(timezone.utc)
    response: Final = gateway.request(
        "GET",
        "/team/daily/activity",
        params={
            "team_ids": team,
            "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        },
    )
    assert response.status_code == 200, response.text
    days: Final = _Report.model_validate_json(response.content).results
    labels: Final = frozenset(chain.from_iterable(day.breakdown.endpoints for day in days))
    return _TeamActivity(
        totals=_total(day.metrics for day in days),
        endpoints={
            label: _total(day.breakdown.endpoints[label].metrics for day in days if label in day.breakdown.endpoints)
            for label in labels
        },
    )


def _requests(count: int) -> Callable[[_Metrics], bool]:
    return lambda totals: totals.api_requests == count


def _settled(gateway: Gateway, team: str, satisfied: Callable[[_Metrics], bool]) -> _TeamActivity:
    held(
        lambda: _activity(gateway, team),
        lambda activity: satisfied(activity.totals),
        holding=3,
        seconds=_SECONDS,
    )
    return _activity(gateway, team)


def _only_under(
    gateway: Gateway, team: str, label: str, satisfied: Callable[[_Metrics], bool] = _requests(1)
) -> _Metrics:
    activity: Final = _settled(gateway, team, satisfied)
    assert set(activity.endpoints) == {label}, activity
    stored: Final = gateway.rows(_DAILY_ROWS, (team,))
    assert {row["endpoint"] for row in stored} == {label}, stored
    return activity.endpoints[label]


def _spend_rows(gateway: Gateway, key: str, count: int = 1) -> list[dict[str, JsonValue]]:
    digest: Final = sha256(key.encode()).hexdigest()
    return eventually(lambda: gateway.rows(_SPEND_ROWS, (digest,)), lambda rows: len(rows) == count, seconds=_SECONDS)


def _assert_usage(metrics: _Metrics, expected: _Metrics) -> None:
    assert metrics.model_dump() == pytest.approx(expected.model_dump()), metrics


def _assert_matches_rows(metrics: _Metrics, rows: list[dict[str, JsonValue]]) -> None:
    assert metrics.spend == pytest.approx(sum(float(str(row["spend"])) for row in rows)), rows
    assert metrics.prompt_tokens == sum(int(str(row["prompt_tokens"])) for row in rows), rows
    assert metrics.completion_tokens == sum(int(str(row["completion_tokens"])) for row in rows), rows


def _assert_one_failure(gateway: Gateway, tenant: _Tenant, label: str) -> None:
    metrics: Final = _only_under(gateway, tenant.team, label)
    rows: Final = _spend_rows(gateway, tenant.key)
    assert rows[0]["status"] == "failure", rows
    assert (metrics.api_requests, metrics.successful_requests, metrics.failed_requests) == (1, 0, 1), metrics
    _assert_matches_rows(metrics, rows)


def _paths(wire: Wire) -> tuple[str, ...]:
    return tuple(urlsplit(request.target).path for request in wire.drain())


def _write_config(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    models: tuple[Mapping[str, JsonValue], ...] = (),
) -> Path:
    base: Final = yaml.safe_load(_BASE_CONFIG.read_text())
    path: Final = directory / "generate-content-attribution.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "model_list": [*base["model_list"], *(dict(model) for model in models)],
                "general_settings": {**base["general_settings"], **(general_settings or {})},
                "litellm_settings": {**base["litellm_settings"], **(litellm_settings or {})},
            }
        )
    )
    return path


@pytest.fixture(scope="module", params=tuple(_ACCOUNTING_SETTINGS), ids=tuple(_ACCOUNTING_SETTINGS))
def accounting_rig(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("generate-content-accounting")
    config: Final = _write_config(directory, general_settings=_ACCOUNTING_SETTINGS[request.param])
    with (
        wire_server(_upstream) as wire,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=config) as proxy,
    ):
        yield _Rig(proxy, wire)


@pytest.fixture(scope="module")
def provider_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("generate-content-providers")
    shutil.copy(_PROVIDER_MODULE, directory / _PROVIDER_MODULE.name)
    with wire_server(_upstream) as wire, gateway_from_environment() as shared:
        config: Final = _write_config(
            directory,
            litellm_settings={
                "custom_provider_map": [
                    {"provider": "studiochat", "custom_handler": "custom_generate_content_provider.CHAT_HANDLER"},
                    {
                        "provider": "studionative",
                        "custom_handler": "custom_generate_content_provider.GENERATE_CONTENT_HANDLER",
                    },
                ]
            },
            models=(
                {"model_name": "studio-chat", "litellm_params": {"model": "studiochat/flare-1", "api_base": wire.url}},
                {
                    "model_name": "studio-native",
                    "litellm_params": {"model": "studionative/flare-1", "api_base": wire.url},
                },
            ),
        )
        overrides: Final = {
            "GEMINI_API_BASE": wire.url,
            "GEMINI_API_KEY": "scripted-gemini-key",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        with owned_proxy(shared, directory, overrides, config=config) as proxy:
            yield _Rig(proxy, wire)


@_EACH_PREFIX
@_STREAMS
@_BACKENDS
def test_generate_content_spend_lands_under_its_endpoint_label(
    gateway: Gateway, backend: _Backend, stream: bool, prefix: str
) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, backend)
        tenant: Final = _tenant(scenario, model)
        response: Final = _generate(gateway, model, tenant.key, stream, prefix=prefix)
        assert response.status_code == 200, response.text
        _assert_usage(_only_under(gateway, tenant.team, _label(stream)), backend.usage())
        assert _paths(wire) == (backend.target(stream),)


@pytest.mark.parametrize(
    ("model", "upstream_path"),
    (("studio-chat", "/complete"), ("studio-native", "/generate")),
    ids=("completion_adapter", "generate_content_hook"),
)
def test_custom_provider_generate_content_lands_under_the_label(
    provider_rig: _Rig, model: str, upstream_path: str
) -> None:
    provider_rig.wire.drain()
    with provider_rig.gateway.scenario() as scenario:
        tenant: Final = _tenant(scenario, model)
        response: Final = _generate(provider_rig.gateway, model, tenant.key, False)
        assert response.status_code == 200, response.text
        metrics: Final = _only_under(provider_rig.gateway, tenant.team, _GENERATE_LABEL)
        assert (metrics.api_requests, metrics.successful_requests, metrics.failed_requests) == (1, 1, 0), metrics
        _assert_matches_rows(metrics, _spend_rows(provider_rig.gateway, tenant.key))
        assert _paths(provider_rig.wire) == (upstream_path,)


@_STREAMS
@pytest.mark.parametrize(
    ("primary", "fallback"),
    ((_NON_GEMINI, _GEMINI), (_GEMINI, _NON_GEMINI)),
    ids=("non_gemini_to_gemini", "gemini_to_non_gemini"),
)
def test_fallback_lands_under_the_requested_routes_label(
    gateway: Gateway, primary: _Backend, fallback: _Backend, stream: bool
) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        failing: Final = _deployment(scenario, wire, primary, fault="/down")
        answering: Final = _deployment(scenario, wire, fallback)
        tenant: Final = _tenant(scenario, failing, answering)
        response: Final = _generate(gateway, failing, tenant.key, stream, fallbacks=[answering])
        assert response.status_code == 200, response.text
        metrics: Final = _only_under(
            gateway, tenant.team, _label(stream), lambda totals: totals.successful_requests == 1
        )
        assert metrics.successful_requests == 1, metrics
        assert metrics.spend == pytest.approx(fallback.usage().spend), metrics
        assert (metrics.prompt_tokens, metrics.completion_tokens) == (
            fallback.prompt_tokens,
            fallback.completion_tokens,
        ), metrics
        paths: Final = _paths(wire)
        assert f"/down{primary.target(stream)}" in paths, paths
        assert paths.count(fallback.target(stream)) == 1, paths


def test_non_gemini_cache_hit_counts_under_the_label(gateway: Gateway) -> None:
    text: Final = f"cached attribution probe {uuid.uuid4().hex}"
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _NON_GEMINI)
        tenant: Final = _tenant(scenario, model)
        responses: Final = tuple(_generate(gateway, model, tenant.key, False, text=text) for _ in range(2))
        assert [response.status_code for response in responses] == [200, 200], responses[-1].text
        metrics: Final = _only_under(gateway, tenant.team, _GENERATE_LABEL, _requests(2))
        rows: Final = _spend_rows(gateway, tenant.key, 2)
        assert sum(row["cache_hit"] == "True" for row in rows) == 1, rows
        assert (metrics.api_requests, metrics.successful_requests, metrics.failed_requests) == (2, 2, 0), metrics
        assert metrics.spend == pytest.approx(_NON_GEMINI.usage().spend), metrics
        _assert_matches_rows(metrics, rows)
        assert _paths(wire) == (_NON_GEMINI.target(False),)


@_STREAMS
@_BACKENDS
def test_upstream_rejection_lands_under_the_label(gateway: Gateway, backend: _Backend, stream: bool) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, backend, fault="/rejected")
        tenant: Final = _tenant(scenario, model)
        response: Final = _generate(gateway, model, tenant.key, stream)
        assert response.status_code == 400, response.text
        _assert_one_failure(gateway, tenant, _label(stream))


@_BACKENDS
def test_mid_stream_drop_lands_under_the_stream_label(gateway: Gateway, backend: _Backend) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, backend, fault="/cut")
        tenant: Final = _tenant(scenario, model)
        response: Final = _generate(gateway, model, tenant.key, True)
        assert response.status_code == 200, response.text
        _assert_one_failure(gateway, tenant, _STREAM_LABEL)


@_STREAMS
@_BACKENDS
def test_model_alias_containing_responses_keeps_the_label(gateway: Gateway, backend: _Backend, stream: bool) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, backend, name=f"team-responses-{uuid.uuid4().hex}")
        tenant: Final = _tenant(scenario, model)
        response: Final = _generate(gateway, model, tenant.key, stream)
        assert response.status_code == 200, response.text
        _assert_usage(_only_under(gateway, tenant.team, _label(stream)), backend.usage())


@_STREAMS
@_BACKENDS
def test_label_survives_accounting_settings(accounting_rig: _Rig, backend: _Backend, stream: bool) -> None:
    proxy: Final = accounting_rig.gateway
    with proxy.scenario() as scenario:
        model: Final = _deployment(scenario, accounting_rig.wire, backend)
        tenant: Final = _tenant(scenario, model)
        response: Final = _generate(proxy, model, tenant.key, stream)
        assert response.status_code == 200, response.text
        _assert_usage(_only_under(proxy, tenant.team, _label(stream)), backend.usage())


@_EACH_PREFIX
@_STREAMS
def test_non_gemini_spend_row_keeps_the_acompletion_call_type(gateway: Gateway, stream: bool, prefix: str) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _NON_GEMINI)
        tenant: Final = _tenant(scenario, model)
        response: Final = _generate(gateway, model, tenant.key, stream, prefix=prefix)
        assert response.status_code == 200, response.text
        rows: Final = _spend_rows(gateway, tenant.key)
        assert rows[0]["call_type"] == "acompletion", rows


@_STREAMS
def test_chat_completions_stay_under_the_chat_label(gateway: Gateway, stream: bool) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _NON_GEMINI)
        tenant: Final = _tenant(scenario, model)
        response: Final = _chat(gateway, model, tenant.key, stream)
        assert response.status_code == 200, response.text
        _assert_usage(_only_under(gateway, tenant.team, _CHAT_LABEL), _NON_GEMINI.usage())


@_STREAMS
def test_forged_route_metadata_on_chat_completions_stays_under_the_chat_label(gateway: Gateway, stream: bool) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _NON_GEMINI)
        tenant: Final = _tenant(scenario, model)
        response: Final = _chat(
            gateway,
            model,
            tenant.key,
            stream,
            metadata=dict(_FORGED_ROUTE_METADATA),
            litellm_metadata=dict(_FORGED_ROUTE_METADATA),
        )
        assert response.status_code == 200, response.text
        _assert_usage(_only_under(gateway, tenant.team, _CHAT_LABEL), _NON_GEMINI.usage())


@_STREAMS
def test_gemini_passthrough_gets_no_generate_content_label(provider_rig: _Rig, stream: bool) -> None:
    provider_rig.wire.drain()
    proxy: Final = provider_rig.gateway
    query: Final = "?alt=sse" if stream else ""
    with proxy.scenario() as scenario:
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        response: Final = proxy.client.post(
            f"/gemini/v1beta/models/{_GEMINI_MODEL}:{_method(stream)}{query}",
            json={"contents": _contents(f"passthrough probe {uuid.uuid4().hex}")},
            headers={"x-goog-api-key": key},
        )
        assert response.status_code == 200, response.text
        activity: Final = _settled(proxy, team, _requests(1))
        assert activity.endpoints == {}, activity
        assert _paths(provider_rig.wire) == (f"/v1beta/models/{_GEMINI_MODEL}:{_method(stream)}",)


@_EACH_PREFIX
@pytest.mark.parametrize(
    ("stream", "route_name"),
    ((False, "agenerate_content"), (True, "agenerate_content_stream")),
    ids=("generate", "stream"),
)
def test_unknown_model_error_keeps_the_route_name(gateway: Gateway, prefix: str, stream: bool, route_name: str) -> None:
    model: Final = f"unknown-model-{uuid.uuid4().hex}"
    response: Final = _generate(gateway, model, gateway.key, stream, prefix=prefix)
    assert response.status_code == 400, response.text
    assert f"{route_name}: Invalid model name passed in model={model}" in response.text, response.text
