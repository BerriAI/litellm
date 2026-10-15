import asyncio
import json
import math
import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows, write_rows
from integration._support.prisma_clients import connected_proxy_prisma
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.constants import PTU_SENTINEL_API_KEY

if TYPE_CHECKING:
    from litellm.llms.azure.ptu_capacity import PTUCapacity

pytestmark: Final = pytest.mark.timeout(4 * graceful_stop_seconds() + 240)

_GROUP: Final = "ptu-ledger"
_SHARED_DEPLOYMENT: Final = "ledger-shared-gpt41"
_OPEN_DEPLOYMENT: Final = "ledger-open-gpt41"
_UPSTREAM_URL_PLACEHOLDER: Final = "upstream-url-placeholder"
_RATE: Final = 2.0
_ATTRIBUTION_ON: Final = MappingProxyType(
    {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True", "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "5"}
)
_ATTRIBUTION_OFF: Final = MappingProxyType({"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "False"})
_PROMPT_TOKENS: Final = 1000
_CACHED_TOKENS: Final = 800
_COMPLETION_TOKENS: Final = 10
_HOLDER_REQUESTS: Final = 2
_CONFIG: Final = TypeAdapter(dict[str, JsonValue])
_DEPLOYMENT_PATH: Final = re.compile(r"/openai/deployments/([^/]+)/")


def _gpt41() -> "PTUCapacity":
    from litellm.llms.azure.ptu_capacity import AZURE_PTU_CAPACITY

    return AZURE_PTU_CAPACITY["gpt-4.1"]


def _reply(request: Request) -> Reply:
    matched: Final = _DEPLOYMENT_PATH.search(request.target)
    deployment: Final = matched.group(1) if matched else "unknown"
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{deployment}-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4.1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "served"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": _PROMPT_TOKENS,
                    "completion_tokens": _COMPLETION_TOKENS,
                    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
                    "prompt_tokens_details": {"cached_tokens": _CACHED_TOKENS},
                },
            }
        ).encode()
    )


def _ptu_terms(shares: Mapping[str, int], effective_from: str) -> dict[str, JsonValue]:
    return {
        "base_model": "azure/gpt-4.1",
        "ptu_count": sum(shares.values()),
        "cost_per_ptu_per_hour": _RATE,
        "ptu_effective_from": effective_from,
        "ptu_shares": dict(shares),
    }


def _azure(
    model_name: str, deployment: str, api_base: str, model_info: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"azure/{deployment}",
            "api_base": api_base,
            "api_key": "synthetic-azure-key",
            "api_version": "2025-03-01-preview",
        },
        "model_info": dict(model_info),
    }


def _config(directory: Path, wire_url: str, holder: str, name: str) -> Path:
    config: Final = {
        **_CONFIG.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())),
        "model_list": [
            _azure(
                _GROUP,
                _SHARED_DEPLOYMENT,
                _UPSTREAM_URL_PLACEHOLDER,
                {"id": f"{_SHARED_DEPLOYMENT}-{uuid.uuid4().hex}", **_ptu_terms({holder: 1}, "2026-01-01T00:00:00Z")},
            ),
            _azure(
                _GROUP,
                _OPEN_DEPLOYMENT,
                _UPSTREAM_URL_PLACEHOLDER,
                {"id": f"{_OPEN_DEPLOYMENT}-{uuid.uuid4().hex}", "base_model": "azure/gpt-4.1"},
            ),
        ],
        "router_settings": {"disable_cooldowns": True, "num_retries": 0, "ignore_invalid_deployments": True},
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config).replace(_UPSTREAM_URL_PLACEHOLDER, wire_url))
    return path


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _window() -> Mapping[str, str]:
    return MappingProxyType(
        {
            "start_date": (_today() - timedelta(days=1)).isoformat(),
            "end_date": (_today() + timedelta(days=1)).isoformat(),
        }
    )


def _team_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))


def _delete_team_once_unheld(proxy: Gateway, team_id: str) -> None:
    if _team_rows(team_id):
        eventually(
            lambda: proxy.request("POST", "/team/delete", {"team_ids": [team_id]}).status_code,
            lambda status: status == 200,
            seconds=60,
        )
    assert _team_rows(team_id) == []


def _delete_user(proxy: Gateway, user_id: str) -> None:
    response: Final = proxy.request("POST", "/user/delete", {"user_ids": [user_id]})
    assert response.status_code == 200, response.text


def _team(proxy: Gateway, cleanups: ExitStack) -> str:
    team_id: Final = f"ptu-ledger-team-{uuid.uuid4().hex}"
    proxy.post("/team/new", {"team_id": team_id, "team_alias": team_id})
    cleanups.callback(_delete_team_once_unheld, proxy, team_id)
    return team_id


def _chat(proxy: Gateway, key: str) -> str:
    response: Final = proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": _GROUP, "max_tokens": 16, "messages": [{"role": "user", "content": uuid.uuid4().hex}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def _model_groups(day: JsonValue) -> dict[str, JsonValue]:
    return object_value(object_value(object_value(day)["breakdown"])["model_groups"])


def _model_group_metrics(activity: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    results: Final = activity["results"]
    assert isinstance(results, list), activity
    return tuple(
        object_value(object_value(_model_groups(day)[_GROUP])["metrics"])
        for day in results
        if _GROUP in _model_groups(day)
    )


def _requests(activity: Mapping[str, JsonValue]) -> int:
    return sum(int(str(metrics["api_requests"])) for metrics in _model_group_metrics(activity))


def _ptu_hours_per_request() -> float:
    capacity: Final = _gpt41()
    normalized: Final = (
        (_PROMPT_TOKENS - _CACHED_TOKENS)
        + capacity.cached_input_ratio * _CACHED_TOKENS
        + capacity.output_to_input_ratio * _COMPLETION_TOKENS
    )
    return normalized / (capacity.input_tpm_per_ptu * 60)


@dataclass(frozen=True, slots=True)
class _Ledger:
    proxy: Gateway
    config: Path
    holder: str
    outsider: str
    member: str
    holder_ids: tuple[str, ...]
    outsider_id: str


def _activity(proxy: Gateway, path: str, params: Mapping[str, str], requests: int) -> dict[str, JsonValue]:
    return eventually(
        lambda: proxy.get(path, {**_window(), **params}),
        lambda activity: _requests(activity) == requests,
        seconds=60,
    )


@pytest.fixture(scope="module")
def ledger(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Ledger]:
    directory: Final = tmp_path_factory.mktemp("ptu-ledger")
    holder: Final = f"ptu-ledger-holder-{uuid.uuid4().hex}"
    outsider: Final = f"ptu-ledger-outsider-{uuid.uuid4().hex}"
    member: Final = f"ptu-ledger-member-{uuid.uuid4().hex}"
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        wire: Final = stack.enter_context(wire_server(_reply))
        config: Final = _config(directory, wire.url, holder, "ptu-ledger")
        stack.callback(_delete_team_once_unheld, gateway, outsider)
        stack.callback(_delete_team_once_unheld, gateway, holder)
        owned: Final = stack.enter_context(
            owned_proxy_process(gateway, directory, _ATTRIBUTION_ON, config=config, workers=2)
        )
        proxy: Final = owned.gateway
        proxy.post("/team/new", {"team_id": holder, "team_alias": holder})
        proxy.post("/team/new", {"team_id": outsider, "team_alias": outsider})
        proxy.post("/user/new", {"user_id": member, "auto_create_key": False})
        stack.callback(_delete_user, proxy, member)
        holder_key: Final = string_value(proxy.post("/key/generate", {"team_id": holder, "user_id": member})["key"])
        outsider_key: Final = string_value(proxy.post("/key/generate", {"team_id": outsider})["key"])
        holder_ids: Final = tuple(_chat(proxy, holder_key) for _ in range(_HOLDER_REQUESTS))
        outsider_id: Final = _chat(proxy, outsider_key)
        _activity(proxy, "/team/daily/activity", {"team_ids": holder}, _HOLDER_REQUESTS)
        _activity(proxy, "/team/daily/activity", {"team_ids": outsider}, 1)
        yield _Ledger(proxy, config, holder, outsider, member, holder_ids, outsider_id)


@pytest.mark.parametrize(
    "path", ["/team/daily/activity", "/team/daily/activity/aggregated"], ids=["daily", "daily-aggregated"]
)
def test_a_share_holder_reads_the_ptu_hours_its_tokens_used(ledger: _Ledger, path: str) -> None:
    assert all(identity.startswith(f"chatcmpl-{_SHARED_DEPLOYMENT}-") for identity in ledger.holder_ids)

    activity: Final = _activity(ledger.proxy, path, {"team_ids": ledger.holder}, _HOLDER_REQUESTS)

    (metrics,) = _model_group_metrics(activity)
    expected: Final = _HOLDER_REQUESTS * _ptu_hours_per_request()
    assert (metrics["prompt_tokens"], metrics["completion_tokens"], metrics["cache_read_input_tokens"]) == (
        _HOLDER_REQUESTS * _PROMPT_TOKENS,
        _HOLDER_REQUESTS * _COMPLETION_TOKENS,
        _HOLDER_REQUESTS * _CACHED_TOKENS,
    )
    assert math.isclose(float(str(metrics["ptu_hours"])), expected), activity
    assert math.isclose(float(str(object_value(activity["metadata"])["total_ptu_hours"])), expected), activity


def test_a_team_served_by_the_open_deployment_reads_no_ptu_hours(ledger: _Ledger) -> None:
    assert ledger.outsider_id.startswith(f"chatcmpl-{_OPEN_DEPLOYMENT}-")

    activity: Final = _activity(ledger.proxy, "/team/daily/activity", {"team_ids": ledger.outsider}, 1)

    (metrics,) = _model_group_metrics(activity)
    assert metrics["prompt_tokens"] == _PROMPT_TOKENS
    assert metrics["ptu_hours"] == 0, activity
    assert object_value(activity["metadata"])["total_ptu_hours"] == 0, activity


def test_user_activity_reports_no_ptu_hours(ledger: _Ledger) -> None:
    activity: Final = _activity(ledger.proxy, "/user/daily/activity", {"user_id": ledger.member}, _HOLDER_REQUESTS)

    (metrics,) = _model_group_metrics(activity)
    assert metrics["prompt_tokens"] == _HOLDER_REQUESTS * _PROMPT_TOKENS
    assert metrics["ptu_hours"] == 0, activity


def test_with_attribution_off_team_activity_reports_no_ptu_hours(ledger: _Ledger, tmp_path: Path) -> None:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, tmp_path, _ATTRIBUTION_OFF, config=ledger.config, workers=2) as owned,
    ):
        activity: Final = _activity(
            owned.gateway, "/team/daily/activity", {"team_ids": ledger.holder}, _HOLDER_REQUESTS
        )

    (metrics,) = _model_group_metrics(activity)
    assert metrics["prompt_tokens"] == _HOLDER_REQUESTS * _PROMPT_TOKENS
    assert metrics["ptu_hours"] == 0, activity
    assert object_value(activity["metadata"])["total_ptu_hours"] == 0, activity


async def _rollup(day: date) -> None:
    from litellm.proxy.spend_tracking.ptu_flat_cost_rollup import run_ptu_flat_cost_rollup

    async with connected_proxy_prisma() as prisma:
        await run_ptu_flat_cost_rollup(prisma, target_date=day)


def _flat_costs(model_id: str, day: date) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT team_id, ptu_flat_cost FROM "LiteLLM_DailyTeamSpend" WHERE api_key = %s AND model = %s AND date = %s',
        (PTU_SENTINEL_API_KEY, model_id, day.isoformat()),
    )
    return {string_value(row["team_id"]): row["ptu_flat_cost"] for row in rows}


def _delete_flat_costs(model_id: str) -> None:
    write_rows(
        'DELETE FROM "LiteLLM_DailyTeamSpend" WHERE api_key = %s AND model = %s', (PTU_SENTINEL_API_KEY, model_id)
    )


def test_the_rollup_bills_each_team_its_share_and_follows_a_move(ledger: _Ledger) -> None:
    yesterday: Final = _today() - timedelta(days=1)
    effective_from: Final = f"{yesterday.isoformat()}T00:00:00Z"
    with ExitStack() as cleanups:
        first: Final = _team(ledger.proxy, cleanups)
        second: Final = _team(ledger.proxy, cleanups)
        third: Final = _team(ledger.proxy, cleanups)
        created: Final = ledger.proxy.post(
            "/model/new",
            _azure(
                f"ptu-ledger-rollup-{uuid.uuid4().hex}",
                "ledger-rollup-gpt41",
                "http://127.0.0.1:9",
                _ptu_terms({first: 2, second: 1}, effective_from),
            ),
        )
        model_id: Final = string_value(object_value(created["model_info"])["id"])
        cleanups.callback(_delete_flat_costs, model_id)
        cleanups.callback(ledger.proxy.post, "/model/delete", {"id": model_id})

        asyncio.run(_rollup(yesterday))
        asyncio.run(_rollup(yesterday))
        assert _flat_costs(model_id, yesterday) == {first: 2 * _RATE * 24, second: 1 * _RATE * 24}

        moved: Final = ledger.proxy.request(
            "PATCH", f"/model/{model_id}/update", {"model_info": {"ptu_shares": {first: 1, third: 2}}}
        )
        assert moved.status_code == 200, moved.text
        asyncio.run(_rollup(yesterday))
        assert _flat_costs(model_id, yesterday) == {first: 1 * _RATE * 24, third: 2 * _RATE * 24}
