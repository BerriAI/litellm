from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    string_value,
)
from integration._support.process import owned_proxy
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

CAP: Final = 0.1
BUDGET_EXCEEDED: Final = 422
FAST_RESET_TICK: Final = MappingProxyType(
    {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "2"}
)


@pytest.fixture(scope="module")
def resetting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as shared,
        owned_proxy(shared, tmp_path_factory.mktemp("fast-reset"), FAST_RESET_TICK) as replica,
    ):
        yield replica


def _policy_config(directory: Path, policy: str) -> Path:
    shared: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    general_settings: Final = {
        **JSON_OBJECT.validate_python(shared["general_settings"]),
        "self_serve_budget_policy": policy,
    }
    path: Final = directory / f"{policy}.yaml"
    path.write_text(yaml.safe_dump({**shared, "general_settings": general_settings}))
    return path


@pytest.fixture(scope="module", params=("lower_only", "ceiling"))
def self_serve(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, Gateway]]:
    policy: Final = string_value(request.param)
    directory: Final = tmp_path_factory.mktemp(policy)
    with (
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=_policy_config(directory, policy)) as replica,
    ):
        yield policy, replica


def _owner_credential(scenario: Scenario) -> str:
    return scenario.key(user_id=scenario.user(user_role="internal_user"))


def _owned_key(gateway: Gateway, scenario: Scenario, owner: str, **fields: JsonValue) -> str:
    response: Final = gateway.request("POST", "/key/generate", fields, key=owner)
    assert response.status_code == 200, response.text
    token: Final = string_value(JSON_OBJECT.validate_json(response.content)["key"])
    scenario.cleanups.callback(scenario.delete_key, token)
    return token


def _key_row(gateway: Gateway, key: str) -> dict[str, JsonValue]:
    rows: Final = gateway.rows(
        "SELECT spend, max_budget, budget_duration, budget_id, budget_reset_at::text AS reset_at"
        ' FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )
    assert len(rows) == 1, rows
    return rows[0]


def _exhaust(gateway: Gateway, key: str) -> None:
    gateway.post("/key/update", {"key": key, "spend": CAP})
    assert _key_row(gateway, key)["spend"] == CAP


def _chat_status(gateway: Gateway, model: str, key: str) -> int:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "budget"}]}, key=key
    ).status_code


def test_key_owner_cannot_shorten_budget_duration_to_restart_an_exhausted_budget(resetting: Gateway) -> None:
    with resetting.scenario() as scenario:
        model: Final = scenario.model()
        owner: Final = _owner_credential(scenario)
        target: Final = _owned_key(resetting, scenario, owner, models=[model], max_budget=CAP, budget_duration="30d")
        control: Final = _owned_key(resetting, scenario, owner, models=[model], max_budget=CAP, budget_duration="30d")
        _exhaust(resetting, target)
        _exhaust(resetting, control)
        assert _chat_status(resetting, model, target) == BUDGET_EXCEEDED
        before: Final = _key_row(resetting, target)

        denied: Final = resetting.request("POST", "/key/update", {"key": target, "budget_duration": "1s"}, key=owner)

        resetting.post("/key/update", {"key": control, "budget_duration": "1s"})
        eventually(lambda: _key_row(resetting, control)["spend"], lambda spend: spend == 0.0, seconds=60)
        assert denied.status_code == 403, denied.text
        assert _key_row(resetting, target) == before
        assert _chat_status(resetting, model, target) == BUDGET_EXCEEDED


def test_key_owner_cannot_swap_or_reschedule_a_tier_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        low: Final = scenario.budget(max_budget=CAP)
        high: Final = scenario.budget(max_budget=100.0)
        owner: Final = _owner_credential(scenario)
        target: Final = _owned_key(gateway, scenario, owner, models=[model], budget_id=low)
        _exhaust(gateway, target)
        assert _chat_status(gateway, model, target) == BUDGET_EXCEEDED
        before: Final = _key_row(gateway, target)

        swapped: Final = gateway.request("POST", "/key/update", {"key": target, "budget_id": high}, key=owner)
        rescheduled: Final = gateway.request("POST", "/key/update", {"key": target, "budget_duration": "1s"}, key=owner)

        assert swapped.status_code == 403, swapped.text
        assert rescheduled.status_code == 403, rescheduled.text
        assert _key_row(gateway, target) == before
        assert _chat_status(gateway, model, target) == BUDGET_EXCEEDED


def test_team_member_cannot_shorten_budget_duration_on_a_team_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_permissions=["/key/update", "/key/info"])
        member: Final = scenario.key(user_id=scenario.member(team), team_id=team)
        target: Final = scenario.key(team_id=team, max_budget=CAP, budget_duration="30d")
        before: Final = _key_row(gateway, target)

        denied: Final = gateway.request("POST", "/key/update", {"key": target, "budget_duration": "1s"}, key=member)

        assert denied.status_code == 403, denied.text
        assert _key_row(gateway, target) == before


def test_key_owner_keeps_edits_that_cannot_restart_a_budget(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owner: Final = _owner_credential(scenario)
        capped: Final = _owned_key(gateway, scenario, owner, max_budget=CAP, budget_duration="monthly")
        uncapped: Final = _owned_key(gateway, scenario, owner)

        renamed: Final = gateway.request("POST", "/key/update", {"key": capped, "key_alias": "renamed"}, key=owner)
        resent: Final = gateway.request("POST", "/key/update", {"key": capped, "budget_duration": "30d"}, key=owner)
        scheduled: Final = gateway.request("POST", "/key/update", {"key": uncapped, "budget_duration": "1d"}, key=owner)

        assert renamed.status_code == 200, renamed.text
        assert resent.status_code == 200, resent.text
        assert scheduled.status_code == 200, scheduled.text
        assert _key_row(gateway, uncapped)["budget_duration"] == "1d"


def test_self_serve_policy_keeps_tier_keys_admin_only(self_serve: tuple[str, Gateway]) -> None:
    _, proxy = self_serve
    with proxy.scenario() as scenario:
        low: Final = scenario.budget(max_budget=CAP)
        high: Final = scenario.budget(max_budget=100.0)
        owner: Final = _owner_credential(scenario)
        target: Final = _owned_key(proxy, scenario, owner, budget_id=low)
        before: Final = _key_row(proxy, target)

        swapped: Final = proxy.request("POST", "/key/update", {"key": target, "budget_id": high}, key=owner)
        rescheduled: Final = proxy.request("POST", "/key/update", {"key": target, "budget_duration": "1d"}, key=owner)

        assert swapped.status_code == 403, swapped.text
        assert rescheduled.status_code == 403, rescheduled.text
        assert _key_row(proxy, target) == before


def test_self_serve_policy_gates_a_new_reset_period_like_a_raised_cap(self_serve: tuple[str, Gateway]) -> None:
    policy, proxy = self_serve
    with proxy.scenario() as scenario:
        owner: Final = _owner_credential(scenario)
        shortened: Final = _owned_key(proxy, scenario, owner, max_budget=CAP, budget_duration="30d")
        cleared: Final = _owned_key(proxy, scenario, owner, max_budget=CAP, budget_duration="30d")

        shorten: Final = proxy.request("POST", "/key/update", {"key": shortened, "budget_duration": "1s"}, key=owner)
        clear: Final = proxy.request("POST", "/key/update", {"key": cleared, "budget_duration": None}, key=owner)

        assert (shorten.status_code, _key_row(proxy, shortened)["budget_duration"]) == (
            (403, "30d") if policy == "lower_only" else (200, "1s")
        ), shorten.text
        assert clear.status_code == 200, clear.text
        assert _key_row(proxy, cleared)["budget_duration"] is None
