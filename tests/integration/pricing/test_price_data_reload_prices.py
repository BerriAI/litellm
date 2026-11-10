import itertools
import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
import pytest
from integration._support.client import Gateway, eventually, held, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import (
    OwnedProxy,
    graceful_stop_seconds,
    owned_proxy,
    owned_proxy_process,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

_ROW: Final = "anthropic.claude-audit-reload-v1:0"
_GEO: Final = f"au.{_ROW}"
_FINETUNE: Final = "ft:gpt-audit-reload-2025-01-01"
_CLAUDE: Final = "claude-audit-reload-sonnet"
_MINI: Final = "gpt-audit-reload-mini"
_FREE: Final = "gpt-audit-reload-free"
_RULE_FREE: Final = "gpt-audit-rule-free"
_FILL_FREE: Final = "gpt-audit-fill-free"
_RULE_UNPRICED: Final = "gpt-audit-rule-unpriced"
_REPORTED_FREE: Final = "gpt-audit-reported-free"
_USAGE: Final = re.compile(r"bill (\d+) (\d+) (\d+)")
_CUSTOM: Final = MappingProxyType({"input_cost_per_token": 1.1e-06, "output_cost_per_token": 2.3e-06})
_COST_MAP_LIMITS: Final = MappingProxyType(
    {"MODEL_COST_MAP_MIN_MODEL_COUNT": "1", "MODEL_COST_MAP_MAX_SHRINK_RATIO": "0"}
)
_PUBLISHED_SCALE: Final = 2.0
_CATALOG: Final[Mapping[str, Mapping[str, JsonValue]]] = MappingProxyType(
    {
        _ROW: {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_read_input_token_cost": 3e-07,
            "input_cost_per_token_above_200k_tokens": 6e-06,
            "output_cost_per_token_above_200k_tokens": 2.25e-05,
        },
        _GEO: {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
            "input_cost_per_token": 3.3e-06,
            "output_cost_per_token": 1.65e-05,
        },
        _FINETUNE: {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 3.75e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_read_input_token_cost": 1.875e-06,
        },
        _CLAUDE: {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_read_input_token_cost": 3e-07,
        },
        _FILL_FREE: {"litellm_provider": "openai", "mode": "chat"},
        _FREE: {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        _MINI: {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 5e-07,
            "output_cost_per_token": 2e-06,
        },
    }
)


@dataclass(frozen=True, slots=True)
class _Published:
    scale: float
    added: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Failing:
    status: int


_Served: TypeAlias = _Published | _Failing


class _CostMapServer:
    def __init__(self, serving: _Served = _Published(1.0)) -> None:
        self.serving: _Served = serving


@dataclass(frozen=True, slots=True)
class _Usage:
    prompt: int
    cached: int
    completion: int

    def text(self) -> str:
        return f"bill {self.prompt} {self.cached} {self.completion}"


@dataclass(frozen=True, slots=True)
class _Rates:
    input: float
    output: float
    cache_read: float

    def cost(self, usage: _Usage) -> float:
        return (
            (usage.prompt - usage.cached) * self.input + usage.cached * self.cache_read + usage.completion * self.output
        )


@dataclass(frozen=True, slots=True)
class _Deployment:
    name: str
    model: str
    prices: Mapping[str, float]
    usage: _Usage
    rates: Callable[[float], _Rates | None]
    reported_cost: float | None = None
    listed_rates: Callable[[float], _Rates | None] | None = None


@dataclass(frozen=True, slots=True)
class _Charge:
    request_id: str
    expected: float


def _catalog(scale: float, added: tuple[str, ...] = ()) -> dict[str, JsonValue]:
    return {
        **{model: {"litellm_provider": "openai", "mode": "chat"} for model in added},
        **{
            model: {name: value * scale if isinstance(value, float) else value for name, value in row.items()}
            for model, row in _CATALOG.items()
        },
        "fallback_generalizations": {
            "rules": [
                {
                    "name": "free-pricing",
                    "pattern": f"({_RULE_FREE}|{_FILL_FREE})",
                    "fill_missing_for_providers": ["openai"],
                    "model_info": {"mode": "chat", "input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
                },
                {"name": "unpriced-capabilities", "pattern": _RULE_UNPRICED, "model_info": {"mode": "chat"}},
            ]
        },
    }


def _rate(model: str, field: str, scale: float) -> float:
    rate: Final = _CATALOG[model][field]
    assert isinstance(rate, float), rate
    return rate * scale


def _catalog_rates(model: str) -> Callable[[float], _Rates]:
    return lambda scale: _Rates(
        input=_rate(model, "input_cost_per_token", scale),
        output=_rate(model, "output_cost_per_token", scale),
        cache_read=_rate(model, "cache_read_input_token_cost", scale)
        if "cache_read_input_token_cost" in _CATALOG[model]
        else _rate(model, "input_cost_per_token", scale),
    )


def _custom_rates(_scale: float) -> _Rates:
    return _Rates(
        input=_CUSTOM["input_cost_per_token"],
        output=_CUSTOM["output_cost_per_token"],
        cache_read=_CUSTOM["input_cost_per_token"],
    )


def _unpriced(_scale: float) -> None:
    return None


def _expected_cost(rates: _Rates | None, usage: _Usage) -> float:
    return 0.0 if rates is None else rates.cost(usage)


def _expected_listing(rates: _Rates | None) -> tuple[object, object]:
    if rates is None:
        return None, None
    return pytest.approx(rates.input), pytest.approx(rates.output)


def _connection(model: str, wire_url: str) -> dict[str, JsonValue]:
    if model.startswith("bedrock/"):
        return {"api_base": wire_url, "api_key": "synthetic-bedrock-bearer", "aws_region_name": "ap-northeast-1"}
    if model.startswith("azure/"):
        return {"api_base": wire_url, "api_key": "synthetic-azure-key", "api_version": "2025-01-01-preview"}
    return {"api_base": f"{wire_url}/v1", "api_key": "synthetic-openai-key"}


def _config(
    directory: Path,
    deployments: tuple[_Deployment, ...],
    wire_url: str,
    general: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    config: Final = directory / f"price_data_reload_config_{uuid.uuid4().hex}.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": deployment.name,
                        "litellm_params": {
                            "model": deployment.model,
                            **_connection(deployment.model, wire_url),
                            **deployment.prices,
                        },
                    }
                    for deployment in deployments
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    **general,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _usage_in(request: Request) -> _Usage:
    found: Final = _USAGE.search(request.body.decode())
    assert found is not None, request
    return _Usage(prompt=int(found.group(1)), cached=int(found.group(2)), completion=int(found.group(3)))


def _reply(request: Request) -> Reply:
    usage: Final = _usage_in(request)
    if "/converse" in request.target:
        return Reply(
            body=json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "priced"}]}},
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": usage.prompt,
                        "outputTokens": usage.completion,
                        "totalTokens": usage.prompt + usage.completion,
                    },
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
        )
    if "/invoke" in request.target:
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-scripted",
                    "content": [{"type": "text", "text": "priced"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage.prompt, "output_tokens": usage.completion},
                }
            ).encode()
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "scripted",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "priced"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": usage.prompt,
                    "completion_tokens": usage.completion,
                    "total_tokens": usage.prompt + usage.completion,
                    "prompt_tokens_details": {"cached_tokens": usage.cached},
                    **({"cost": 0.0} if object_value(json.loads(request.body))["model"] == _REPORTED_FREE else {}),
                },
            }
        ).encode()
    )


def _cost_map_reply(served: _Served) -> Reply:
    match served:
        case _Published(scale=scale, added=added):
            return Reply(body=json.dumps(_catalog(scale, added)).encode())
        case _Failing(status=status):
            return Reply(status=status, body=b"{}")


def _responder(server: _CostMapServer) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/model_prices.json":
            return _cost_map_reply(server.serving)
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        return _reply(request)

    return respond


def _listed_rates(proxy: Gateway, name: str) -> tuple[JsonValue, JsonValue]:
    entries: Final = proxy.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    info: Final = next(
        object_value(object_value(entry)["model_info"])
        for entry in entries
        if object_value(entry)["model_name"] == name
    )
    return info.get("input_cost_per_token"), info.get("output_cost_per_token")


def _billed(client: httpx.Client, key: str, deployment: _Deployment) -> tuple[str, float | None]:
    response: Final = client.post(
        "/v1/chat/completions",
        json={"model": deployment.name, "messages": [{"role": "user", "content": deployment.usage.text()}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert response.status_code == 200, response.text
    reported: Final = response.headers.get("x-litellm-response-cost")
    return string_value(object_value(response.json())["id"]), None if reported is None else float(reported)


def _charged_as_listed(proxy: Gateway, deployment: _Deployment, scale: float) -> _Charge:
    rates: Final = deployment.rates(scale)
    assert _listed_rates(proxy, deployment.name) == _expected_listing(
        deployment.listed_rates(scale) if deployment.listed_rates is not None else rates
    ), deployment.name
    request_id, charged = _billed(proxy.client, proxy.key, deployment)
    expected: Final = (
        deployment.reported_cost if deployment.reported_cost is not None else _expected_cost(rates, deployment.usage)
    )
    assert charged == (
        None if rates is None and deployment.reported_cost is None else pytest.approx(expected, rel=1e-6)
    ), deployment.name
    return _Charge(request_id=request_id, expected=expected)


def _spend(request_id: str) -> float:
    rows: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float)), rows
    return float(spend)


def _proxy_environment(wire_url: str) -> dict[str, str]:
    return {**_COST_MAP_LIMITS, "LITELLM_MODEL_COST_MAP_URL": f"{wire_url}/model_prices.json"}


def _assert_priced_across_a_reload(gateway: Gateway, tmp_path: Path, deployments: tuple[_Deployment, ...]) -> None:
    server: Final = _CostMapServer()
    with (
        wire_server(_responder(server)) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            _proxy_environment(wire.url),
            config=_config(tmp_path, deployments, wire.url),
            remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",),
        ) as proxy,
    ):
        before: Final = tuple(_charged_as_listed(proxy, deployment, 1.0) for deployment in deployments)
        server.serving = _Published(_PUBLISHED_SCALE)
        reload: Final = proxy.request("POST", "/reload/model_cost_map")
        assert reload.status_code == 200, reload.text
        after: Final = tuple(_charged_as_listed(proxy, deployment, _PUBLISHED_SCALE) for deployment in deployments)
        for charge in (*before, *after):
            assert _spend(charge.request_id) == pytest.approx(charge.expected, rel=1e-6)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_custom_prices_on_one_deployment_never_reach_a_sibling_on_the_same_backend_across_a_reload(
    gateway: Gateway, tmp_path: Path
) -> None:
    _assert_priced_across_a_reload(
        gateway,
        tmp_path,
        (
            _Deployment("mini-custom", f"openai/{_MINI}", _CUSTOM, _Usage(1000, 0, 500), _custom_rates),
            _Deployment("mini-catalog", f"openai/{_MINI}", {}, _Usage(1000, 0, 500), _catalog_rates(_MINI)),
        ),
    )


def _catalog_tier_rates(model: str) -> Callable[[float], _Rates]:
    return lambda scale: _Rates(
        input=_rate(model, "input_cost_per_token_above_200k_tokens", scale),
        output=_rate(model, "output_cost_per_token_above_200k_tokens", scale),
        cache_read=_rate(model, "cache_read_input_token_cost", scale),
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_every_deployment_bills_the_catalog_row_its_registration_resolves_to_across_a_reload(
    gateway: Gateway, tmp_path: Path
) -> None:
    _assert_priced_across_a_reload(
        gateway,
        tmp_path,
        (
            _Deployment(
                "region-unpriced", f"bedrock/ap-northeast-1.{_ROW}", {}, _Usage(1000, 0, 500), _catalog_rates(_ROW)
            ),
            _Deployment(
                "finetune-unpriced",
                f"openai/{_FINETUNE}:org:suffix:y",
                {},
                _Usage(2000, 1000, 1000),
                _catalog_rates(_FINETUNE),
            ),
            _Deployment(
                "finetune-custom", f"openai/{_FINETUNE}:org:suffix:z", _CUSTOM, _Usage(2000, 0, 1000), _custom_rates
            ),
            _Deployment("cross-provider-unpriced", f"azure/{_CLAUDE}", {}, _Usage(2000, 1000, 1000), _unpriced),
            _Deployment("catalog-free", f"openai/{_FREE}", {}, _Usage(2000, 1000, 1000), _catalog_rates(_FREE)),
            _Deployment("rule-free", f"openai/{_RULE_FREE}", {}, _Usage(2000, 1000, 1000), _catalog_rates(_FREE)),
            _Deployment(
                "fill-rule-free",
                f"openai/{_FILL_FREE}",
                {},
                _Usage(2000, 1000, 1000),
                _catalog_rates(_FREE),
                listed_rates=_unpriced,
            ),
            _Deployment("rule-unpriced", f"openai/{_RULE_UNPRICED}", {}, _Usage(2000, 1000, 1000), _unpriced),
            _Deployment(
                "provider-reported-free", f"openrouter/{_REPORTED_FREE}", {}, _Usage(2000, 1000, 1000), _unpriced, 0.0
            ),
            _Deployment(
                "cross-provider-free",
                f"azure/{_CLAUDE}",
                {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
                _Usage(2000, 1000, 1000),
                _catalog_rates(_FREE),
            ),
            _Deployment(
                "region-custom-above-threshold",
                f"bedrock/ap-northeast-1.{_ROW}",
                _CUSTOM,
                _Usage(250_000, 0, 1000),
                _custom_rates,
            ),
            _Deployment(
                "region-catalog-above-threshold",
                f"bedrock/ap-northeast-1.{_ROW}",
                {},
                _Usage(250_000, 0, 1000),
                _catalog_tier_rates(_ROW),
                listed_rates=_catalog_rates(_ROW),
            ),
        ),
    )


def _storm_deployment(index: int) -> _Deployment:
    prices: Final = MappingProxyType(
        {"input_cost_per_token": (index + 1) * 1e-07, "output_cost_per_token": (index + 1) * 3e-07}
    )
    return _Deployment(
        f"storm-{index}",
        f"openai/gpt-audit-reload-storm-{index}",
        prices,
        _Usage(100, 0, 50),
        lambda _scale: _Rates(
            input=prices["input_cost_per_token"],
            output=prices["output_cost_per_token"],
            cache_read=prices["input_cost_per_token"],
        ),
    )


def _storm_traffic(
    base_url: str, key: str, deployments: tuple[_Deployment, ...], offset: int, finished: threading.Event
) -> tuple[tuple[float, float], ...]:
    rotation: Final = (
        deployments[(offset + turn) % len(deployments)] for turn, _ in enumerate(iter(finished.is_set, True))
    )
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as client:
        return tuple(
            (_billed(client, key, deployment)[1], _expected_cost(deployment.rates(1.0), deployment.usage))
            for deployment in rotation
        )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_price_data_reloads_under_custom_priced_traffic_keep_every_deployment_priced(
    gateway: Gateway, tmp_path: Path
) -> None:
    storm: Final = tuple(_storm_deployment(index) for index in range(24))
    geo: Final = _Deployment("geo-unpriced", f"bedrock/{_GEO}", {}, _Usage(1000, 0, 500), _catalog_rates(_GEO))
    finished: Final = threading.Event()
    with (
        wire_server(_responder(_CostMapServer())) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            _proxy_environment(wire.url),
            config=_config(tmp_path, (geo, *storm), wire.url),
            remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",),
        ) as proxy,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        traffic: Final = tuple(
            pool.submit(_storm_traffic, str(proxy.client.base_url), proxy.key, storm, offset, finished)
            for offset in range(8)
        )
        reloads: Final = tuple(proxy.request("POST", "/reload/model_cost_map") for _ in range(40))
        finished.set()
        charges: Final = tuple(itertools.chain.from_iterable(calls.result() for calls in traffic))
        assert tuple(reload.status_code for reload in reloads) == (200,) * 40, tuple(
            reload.text for reload in reloads if reload.status_code != 200
        )
        assert len(charges) >= 8
        assert tuple(charged for charged, _ in charges) == pytest.approx(tuple(expected for _, expected in charges))
        settled: Final = tuple(_charged_as_listed(proxy, deployment, 1.0) for deployment in (geo, *storm))
        for charge in settled:
            assert _spend(charge.request_id) == pytest.approx(charge.expected, rel=1e-6)


_GEO_CATALOG: Final = _Deployment("geo-unpriced", f"bedrock/{_GEO}", {}, _Usage(1000, 0, 500), _catalog_rates(_GEO))
_POLL_SECONDS: Final = 1
_LIFECYCLE_SETTINGS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "proxy_config_reload_interval_seconds": _POLL_SECONDS,
    }
)


def _lifecycle_proxy(
    gateway: Gateway,
    tmp_path: Path,
    wire_url: str,
    deployments: tuple[_Deployment, ...],
    database: Mapping[str, str] = MappingProxyType({}),
) -> AbstractContextManager[OwnedProxy]:
    return owned_proxy_process(
        gateway,
        tmp_path,
        {**_proxy_environment(wire_url), **database},
        config=_config(tmp_path, deployments, wire_url, _LIFECYCLE_SETTINGS),
        remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",),
    )


def _bill(proxy: Gateway, deployment: _Deployment) -> float | None:
    return _billed(proxy.client, proxy.key, deployment)[1]


def _reload(proxy: Gateway) -> httpx.Response:
    return proxy.request("POST", "/reload/model_cost_map")


def _billed_at(deployment: _Deployment, scale: float) -> Callable[[float | None], bool]:
    expected: Final = _expected_cost(deployment.rates(scale), deployment.usage)
    return lambda charged: charged == pytest.approx(expected, rel=1e-6)


@pytest.mark.timeout(4 * graceful_stop_seconds() + 180)
def test_price_data_changes_only_through_a_successful_reload_or_a_restart(gateway: Gateway, tmp_path: Path) -> None:
    server: Final = _CostMapServer()
    with wire_server(_responder(server)) as wire, scratch_database("reload_lifecycle") as database_url:
        database: Final = {"DATABASE_URL": database_url}
        with _lifecycle_proxy(gateway, tmp_path, wire.url, (_GEO_CATALOG,), database=database) as first:
            _charged_as_listed(first.gateway, _GEO_CATALOG, 1.0)
            server.serving = _Published(_PUBLISHED_SCALE)
            held(
                lambda: _bill(first.gateway, _GEO_CATALOG),
                _billed_at(_GEO_CATALOG, 1.0),
                holding=3 * _POLL_SECONDS,
            )
            server.serving = _Failing(404)
            failed: Final = _reload(first.gateway)
            assert failed.status_code == 502, failed.text
            _charged_as_listed(first.gateway, _GEO_CATALOG, 1.0)
            server.serving = _Published(_PUBLISHED_SCALE)
        with _lifecycle_proxy(gateway, tmp_path, wire.url, (_GEO_CATALOG,), database=database) as restarted:
            _charged_as_listed(restarted.gateway, _GEO_CATALOG, _PUBLISHED_SCALE)


def _add_model(proxy: Gateway, deployment: _Deployment, wire_url: str) -> None:
    created: Final = proxy.post(
        "/model/new",
        {
            "model_name": deployment.name,
            "litellm_params": {
                "model": deployment.model,
                **_connection(deployment.model, wire_url),
                **deployment.prices,
            },
            "model_info": {},
        },
    )
    assert object_value(created["model_info"])["id"], created


def _custom_with_catalog_cache(model: str) -> Callable[[float], _Rates]:
    return lambda scale: _Rates(
        input=_CUSTOM["input_cost_per_token"],
        output=_CUSTOM["output_cost_per_token"],
        cache_read=_rate(model, "cache_read_input_token_cost", scale),
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 180)
def test_a_model_added_through_the_api_keeps_its_prices_and_inherits_the_reloaded_ones(
    gateway: Gateway, tmp_path: Path
) -> None:
    added: Final = _Deployment(
        "added-finetune-custom",
        f"openai/{_FINETUNE}",
        _CUSTOM,
        _Usage(2000, 1000, 1000),
        _custom_with_catalog_cache(_FINETUNE),
    )
    server: Final = _CostMapServer()
    with (
        wire_server(_responder(server)) as wire,
        _lifecycle_proxy(gateway, tmp_path, wire.url, (_GEO_CATALOG,)) as owned,
    ):
        proxy: Final = owned.gateway
        _add_model(proxy, added, wire.url)
        before: Final = tuple(_charged_as_listed(proxy, deployment, 1.0) for deployment in (_GEO_CATALOG, added))
        server.serving = _Published(_PUBLISHED_SCALE)
        reload: Final = _reload(proxy)
        assert reload.status_code == 200, reload.text
        after: Final = tuple(
            _charged_as_listed(proxy, deployment, _PUBLISHED_SCALE) for deployment in (_GEO_CATALOG, added)
        )
        for charge in (*before, *after):
            assert _spend(charge.request_id) == pytest.approx(charge.expected, rel=1e-6)


_WILDCARD: Final = _Deployment("openai/*", "openai/*", {}, _Usage(1000, 0, 500), _unpriced)
_LISTED_AFTER_RELOAD: Final = "gpt-audit-reload-listed"


def _listed_models(proxy: Gateway) -> list[str]:
    data: Final = proxy.get("/v1/models")["data"]
    assert isinstance(data, list), data
    return [string_value(object_value(entry)["id"]) for entry in data]


def _deployment_ids(proxy: Gateway) -> list[str]:
    entries: Final = proxy.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    return [string_value(object_value(object_value(entry)["model_info"])["id"]) for entry in entries]


@pytest.mark.timeout(2 * graceful_stop_seconds() + 180)
def test_a_wildcard_lists_the_models_a_reload_adds_and_never_a_deployment_id(gateway: Gateway, tmp_path: Path) -> None:
    custom: Final = _Deployment("listed-custom", f"openai/{_MINI}", _CUSTOM, _Usage(1000, 0, 500), _custom_rates)
    server: Final = _CostMapServer()
    with (
        wire_server(_responder(server)) as wire,
        _lifecycle_proxy(gateway, tmp_path, wire.url, (_WILDCARD, custom)) as owned,
    ):
        proxy: Final = owned.gateway
        _charged_as_listed(proxy, custom, 1.0)
        assert f"openai/{_LISTED_AFTER_RELOAD}" not in _listed_models(proxy)
        server.serving = _Published(1.0, added=(_LISTED_AFTER_RELOAD,))
        reload: Final = _reload(proxy)
        assert reload.status_code == 200, reload.text
        _charged_as_listed(proxy, custom, 1.0)
        listed: Final = _listed_models(proxy)
        deployment_ids: Final = _deployment_ids(proxy)
        assert f"openai/{_LISTED_AFTER_RELOAD}" in listed, listed
        assert [model for model in listed if any(id_ in model for id_ in deployment_ids)] == [], listed
