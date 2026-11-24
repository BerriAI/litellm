import json
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import isclose
from typing import Final

import httpx
import psycopg
import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.types.proxy.management_endpoints.app_groups import AppGroup
from litellm.types.proxy.management_endpoints.apps import (
    AppDefinition,
    AppResource,
    AppResourceType,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import BudgetGroups
from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    delete_key_if_present,
    eventually,
    string_value,
)
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

_APPS: Final = "/management/v1/apps"
_CUSTOMERS: Final = "/management/v1/customer_organizations"
_MEMBERS: Final = "/management/v1/customer_members"
_TEAMS: Final = "/management/v1/customer_teams"
_PROFILES: Final = "/management/v1/customer_organization_budget_groups"
_COST: Final = 0.06


@pytest.fixture(scope="module", params=tuple(AppResourceType))
def app_gateway(
    session_gateway: Gateway,
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("app-customer-budgets")
    resource_type: Final = TypeAdapter(AppResourceType).validate_python(
        request.param  # pyright: ignore[reportAny]  # pytest parameters are untyped and validated at this boundary
    )
    with owned_proxy(session_gateway, directory, {}, workers=2) as candidate:
        unset: Final = candidate.request("GET", f"{_APPS}/definition")
        assert AppDefinition.model_validate_json(unset.content).resource_type is None
        configured: Final = candidate.request("PUT", f"{_APPS}/definition", {"resource_type": resource_type.value})
        assert configured.status_code == 200, configured.text
        yield candidate


def _reply(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "scripted",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "scripted response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 20,
                    "total_tokens": 40,
                },
            }
        ).encode()
    )


def _create(scenario: Scenario, collection: str, body: Mapping[str, JsonValue], identity: str) -> None:
    response: Final = scenario.gateway.request("POST", collection, body)
    assert response.status_code == 201, response.text
    scenario.cleanups.callback(scenario.gateway.request, "DELETE", f"{collection}/{identity}")


@dataclass(frozen=True, slots=True)
class Customer:
    organization: str
    alias: str
    member: str
    other_member: str


def _customer(scenario: Scenario) -> Customer:
    suffix: Final = uuid.uuid4().hex
    organization, team, member, other = (
        f"app-org-{suffix}",
        f"app-team-{suffix}",
        f"app-member-{suffix}",
        f"app-other-{suffix}",
    )
    _create(
        scenario,
        _CUSTOMERS,
        {
            "customer_org_id": organization,
            "alias": organization,
            "budget": {"max_budget": 10},
            "default_budgets": [{"name": "member-default", "max_budget": 10}],
            "default_budget_template": "member-default",
        },
        organization,
    )
    _create(
        scenario,
        _TEAMS,
        {"customer_team_id": team, "parent_org_id": organization},
        team,
    )
    for identity in (member, other):
        _create(
            scenario,
            _MEMBERS,
            {"customer_member_id": identity, "parent_team_ids": [team]},
            identity,
        )
    return Customer(organization, organization, member, other)


def _rotatable_key(scenario: Scenario, **fields: JsonValue) -> str:
    key: Final = string_value(scenario.gateway.post("/key/generate", fields)["key"])
    scenario.cleanups.callback(delete_key_if_present, scenario.gateway, key)
    return key


def _app(scenario: Scenario, model: str, name: str, *, rotatable: bool = False) -> tuple[str, str]:
    alias: Final = f"{name}-{uuid.uuid4().hex}"
    definition: Final = scenario.gateway.request("GET", f"{_APPS}/definition")
    resource_type: Final = AppDefinition.model_validate_json(definition.content).resource_type
    team: Final = scenario.team(team_alias=alias, models=[model])
    project: Final = (
        scenario.project(team, project_alias=alias, models=[model])
        if resource_type is AppResourceType.PROJECT
        else None
    )
    key_fields: Final[dict[str, JsonValue]] = {
        "team_id": team,
        "project_id": project,
        "models": [model],
        "key_alias": alias,
    }
    key: Final = _rotatable_key(scenario, **key_fields) if rotatable else scenario.key(**key_fields)
    discovered: Final = scenario.gateway.request("GET", f"{_APPS}/resources")
    assert discovered.status_code == 200, discovered.text
    resources: Final = TypeAdapter(tuple[AppResource, ...]).validate_json(discovered.content)
    resource: Final = next(row for row in resources if row.alias == alias)
    assert resource.resource_type is resource_type
    bound: Final = scenario.gateway.request("POST", _APPS, {"resource_id": resource.resource_id})
    assert bound.status_code in (200, 201), bound.text
    app_id: Final = string_value(JSON_OBJECT.validate_json(bound.content)["app_id"])
    scenario.cleanups.callback(scenario.gateway.request, "DELETE", f"{_APPS}/{app_id}")
    return app_id, key


def _budget(gateway: Gateway, app_id: str, customer: Customer, maximum: float) -> None:
    response: Final = gateway.request(
        "PUT",
        f"{_APPS}/{app_id}/customer_organizations/{customer.organization}",
        {
            "budget_override": {"max_budget": maximum},
            "customer_member_budget": {"max_budget": 10},
        },
    )
    assert response.status_code == 200, response.text


def _profile(gateway: Gateway, catalog: str = _PROFILES) -> str:
    current: Final = BudgetGroups.model_validate_json(gateway.request("GET", catalog).content)
    name: Final = f"profile-{uuid.uuid4().hex}"
    response: Final = gateway.request(
        "PATCH",
        catalog,
        {
            "expected_revision": current.budget_groups_revision,
            "default_budget_template": None,
            "default_budgets": [
                *({"budget_id": group.budget_id, "name": group.name} for group in current.default_budgets),
                {"name": name, "max_budget": 0.05, "budget_duration": "1mo"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    groups: Final = BudgetGroups.model_validate_json(response.content)
    return next(group.budget_id for group in groups.default_budgets if group.name == name)


@pytest.mark.parametrize("scope", ("member", "organization"))
def test_member_template_assignment_and_app_update_both_commit(app_gateway: Gateway, scope: str) -> None:
    with app_gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=2) as pool:
        customer: Final = _customer(scenario)
        app_id, _ = _app(scenario, "all-proxy-models", "member-template")
        profile: Final = _profile(app_gateway, "/management/v1/app_customer_member_budget_groups")
        path, field = (
            (f"customer_members/{customer.member}", "budget_id")
            if scope == "member"
            else (f"customer_organizations/{customer.organization}", "customer_member_budget_id")
        )
        assert app_gateway.database_url is not None
        with psycopg.connect(app_gateway.database_url) as blocking:
            blocking.execute(
                'SELECT 1 FROM "LiteLLM_Config" WHERE param_name = %s FOR UPDATE', ("customer_app_definition",)
            )
            waiters: Final = lambda: app_gateway.rows(
                "WITH RECURSIVE blocked(pid) AS (SELECT pid FROM pg_stat_activity "
                "WHERE %s::int = ANY(pg_blocking_pids(pid)) UNION SELECT a.pid FROM pg_stat_activity a "
                "JOIN blocked b ON b.pid = ANY(pg_blocking_pids(a.pid))) SELECT count(*)::int AS waiting FROM blocked",
                (str(blocking.info.backend_pid),),
            )[0]["waiting"]
            updating: Final = pool.submit(
                app_gateway.request, "PATCH", f"{_APPS}/{app_id}", {"default_customer_member_budget_id": profile}
            )
            eventually(waiters, lambda count: count == 1)
            assigning: Final = pool.submit(app_gateway.request, "PUT", f"{_APPS}/{app_id}/{path}", {field: profile})
            eventually(waiters, lambda count: count == 2)
            blocking.commit()
        for response in (updating.result(timeout=15), assigning.result(timeout=15)):
            assert response.status_code == 200, response.text
        saved: Final = app_gateway.request("GET", f"{_APPS}/{app_id}/{path}")
        assert saved.status_code == 200 and JSON_OBJECT.validate_json(saved.content)[field] == profile, saved.text


def _chat(
    gateway: Gateway,
    key: str,
    model: str,
    customer: Customer,
    *,
    member: str | None = None,
    asserted_app: str | None = None,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"app budget {uuid.uuid4().hex}"}],
            "max_tokens": 1,
        },
        key=key,
        headers={
            "x-litellm-customer-org-alias": customer.alias,
            "x-litellm-customer-member-id": member or customer.member,
            **({"x-litellm-app": asserted_app} if asserted_app is not None else {}),
        },
    )


def _blocked(response: httpx.Response) -> None:
    assert response.status_code in (422, 429), response.text
    error: Final = JSON_OBJECT.validate_python(JSON_OBJECT.validate_json(response.content)["error"])
    assert error["type"] == "budget_exceeded", response.text


def _spend(row: Mapping[str, JsonValue]) -> float:
    return TypeAdapter(float).validate_python(row["spend"])


def _blocked_on_every_worker(gateway: Gateway, key: str, model: str, customer: Customer) -> None:
    def worker_id(_: int) -> str | None:
        with httpx.Client(
            base_url=str(gateway.client.base_url),
            timeout=15,
            trust_env=False,
            limits=httpx.Limits(max_connections=1),
        ) as client:
            pinned: Final = replace(gateway, client=client)
            before: Final = pinned.request("GET", "/health/integration-readiness")
            assert before.status_code == 200, before.text
            _blocked(_chat(pinned, key, model, customer))
            after: Final = pinned.request("GET", "/health/integration-readiness")
            assert after.status_code == 200, after.text
            identity: Final = before.headers["x-integration-worker-pid"]
            return identity if after.headers["x-integration-worker-pid"] == identity else None

    def worker_ids() -> frozenset[str]:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return frozenset(identity for identity in pool.map(worker_id, range(8)) if identity is not None)

    eventually(worker_ids, lambda identities: len(identities) == gateway.workers)


def test_app_and_global_customer_budgets_enforce_independently_and_account_once(
    app_gateway: Gateway,
) -> None:
    with wire_server(_reply) as wire, app_gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=f"{wire.url}/v1",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        customer: Final = _customer(scenario)
        profile: Final = _profile(app_gateway)
        chat_app, chat_key = _app(scenario, model, "chat")
        search_app, search_key = _app(scenario, model, "search")
        _budget(app_gateway, chat_app, customer, 10)
        assigned: Final = app_gateway.request(
            "PUT",
            f"{_APPS}/{search_app}/customer_organizations/{customer.organization}",
            {"tier_budget_id": profile, "customer_member_budget": {"max_budget": 10}},
        )
        assert assigned.status_code == 200, assigned.text

        assert _chat(app_gateway, chat_key, model, customer, asserted_app=search_app).status_code == 403
        assert _chat(app_gateway, search_key, model, customer).status_code == 200
        eventually(
            lambda: app_gateway.rows(
                'SELECT spend FROM "LiteLLM_AppCustomerOrgTable" WHERE app_id=%s AND customer_org_id=%s',
                (search_app, customer.organization),
            ),
            lambda rows: len(rows) == 1 and _spend(rows[0]) >= _COST,
            seconds=70,
        )
        wire.drain()
        _blocked_on_every_worker(app_gateway, search_key, model, customer)
        assert not tuple(request for request in wire.drain() if request.method == "POST")
        assert _chat(app_gateway, chat_key, model, customer).status_code == 200

        daily: Final = eventually(
            lambda: app_gateway.rows(
                'SELECT app_id, SUM(spend) AS spend FROM "LiteLLM_DailyCustomerMemberSpend" '
                "WHERE customer_org_id=%s GROUP BY app_id HAVING SUM(spend)>0 ORDER BY app_id",
                (customer.organization,),
            ),
            lambda rows: len(rows) == 2 and sum(_spend(row) for row in rows) >= 2 * _COST,
            seconds=70,
        )
        assert {string_value(row["app_id"]) for row in daily} == {chat_app, search_app}
        assert all(isclose(_spend(row), _COST) for row in daily)
        totals: Final = app_gateway.rows(
            'SELECT app_id, spend FROM "LiteLLM_AppTable" WHERE app_id IN (%s, %s)',
            (chat_app, search_app),
        )
        organization_totals: Final = app_gateway.rows(
            'SELECT app_id, spend FROM "LiteLLM_AppCustomerOrgTable" WHERE customer_org_id=%s',
            (customer.organization,),
        )
        member_totals: Final = app_gateway.rows(
            'SELECT app_id, spend FROM "LiteLLM_AppCustomerMemberTable" WHERE customer_member_id=%s',
            (customer.member,),
        )
        for projection in (totals, organization_totals, member_totals):
            assert {string_value(row["app_id"]) for row in projection} == {
                chat_app,
                search_app,
            }
            assert all(isclose(_spend(row), _COST) for row in projection)
        global_rows: Final = app_gateway.rows(
            'SELECT spend FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id=%s',
            (customer.organization,),
        )
        assert isclose(_spend(global_rows[0]), 2 * _COST)
        member_rows: Final = app_gateway.rows(
            'SELECT spend FROM "LiteLLM_CustomerMemberTable" WHERE customer_member_id=%s',
            (customer.member,),
        )
        assert isclose(_spend(member_rows[0]), 2 * _COST)

        lowered: Final = app_gateway.request(
            "PATCH",
            f"{_CUSTOMERS}/{customer.organization}",
            {"budget": {"max_budget": 2 * _COST}},
        )
        assert lowered.status_code == 200, lowered.text
        wire.drain()
        for key in (chat_key, search_key):
            _blocked_on_every_worker(app_gateway, key, model, customer)
        assert not tuple(request for request in wire.drain() if request.method == "POST")


def test_member_override_blocks_only_that_member_within_its_app(
    app_gateway: Gateway,
) -> None:
    with wire_server(_reply) as wire, app_gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=f"{wire.url}/v1",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        customer: Final = _customer(scenario)
        chat_app, chat_key = _app(scenario, model, "chat")
        search_app, search_key = _app(scenario, model, "search")
        _budget(app_gateway, chat_app, customer, 10)
        _budget(app_gateway, search_app, customer, 10)
        override: Final = app_gateway.request(
            "PUT",
            f"{_APPS}/{search_app}/customer_members/{customer.member}",
            {"budget_override": {"max_budget": 0.05}},
        )
        assert override.status_code == 200, override.text
        assert _chat(app_gateway, search_key, model, customer).status_code == 200
        eventually(
            lambda: app_gateway.rows(
                'SELECT spend FROM "LiteLLM_AppCustomerMemberTable" WHERE app_id=%s AND customer_member_id=%s',
                (search_app, customer.member),
            ),
            lambda rows: len(rows) == 1 and _spend(rows[0]) >= _COST,
            seconds=70,
        )
        _blocked_on_every_worker(app_gateway, search_key, model, customer)
        assert _chat(app_gateway, chat_key, model, customer).status_code == 200
        assert _chat(app_gateway, search_key, model, customer, member=customer.other_member).status_code == 200


@pytest.mark.parametrize("app_gateway", (AppResourceType.VIRTUAL_KEY,), indirect=True)
def test_rotating_a_virtual_key_keeps_its_customer_app_budget_and_spend(
    app_gateway: Gateway,
) -> None:
    with wire_server(_reply) as wire, app_gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=f"{wire.url}/v1",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        customer: Final = _customer(scenario)
        app_id, key = _app(scenario, model, "rotating", rotatable=True)
        _budget(app_gateway, app_id, customer, 0.05)
        assert _chat(app_gateway, key, model, customer).status_code == 200
        eventually(
            lambda: app_gateway.rows(
                'SELECT spend FROM "LiteLLM_AppCustomerOrgTable" WHERE app_id=%s AND customer_org_id=%s',
                (app_id, customer.organization),
            ),
            lambda rows: len(rows) == 1 and _spend(rows[0]) >= _COST,
            seconds=70,
        )
        rotated: Final = app_gateway.request("POST", "/key/regenerate", {"key": key, "grace_period": "1h"})
        assert rotated.status_code == 200, rotated.text
        replacement: Final = string_value(JSON_OBJECT.validate_json(rotated.content)["key"])
        scenario.cleanups.callback(app_gateway.request, "POST", "/key/delete", {"keys": [replacement]})
        _blocked_on_every_worker(app_gateway, replacement, model, customer)
        _blocked_on_every_worker(app_gateway, key, model, customer)
        stored: Final = app_gateway.rows(
            'SELECT spend FROM "LiteLLM_AppCustomerOrgTable" WHERE app_id=%s AND customer_org_id=%s',
            (app_id, customer.organization),
        )
        assert isclose(_spend(stored[0]), _COST)


def _group(scenario: Scenario, budget: Mapping[str, JsonValue]) -> str:
    from psycopg import sql

    response: Final = scenario.gateway.request(
        "POST", "/management/v1/app_groups", {"name": f"group-{uuid.uuid4().hex}", "budget": dict(budget)}
    )
    assert response.status_code == 201, response.text
    identity: Final = string_value(JSON_OBJECT.validate_json(response.content)["app_group_id"])

    def cleanup() -> None:
        assert scenario.gateway.database_url is not None
        with psycopg.connect(scenario.gateway.database_url) as connection:
            connection.execute(
                sql.SQL('UPDATE "LiteLLM_AppTable" SET app_group_id=NULL WHERE app_group_id=%s'), (identity,)
            )
            connection.execute(sql.SQL('DELETE FROM "LiteLLM_AppGroupTable" WHERE app_group_id=%s'), (identity,))

    scenario.cleanups.callback(cleanup)
    return identity


def _group_spend(gateway: Gateway, group_id: str, expected: float) -> None:
    rows: Final = eventually(
        lambda: gateway.rows('SELECT spend FROM "LiteLLM_AppGroupTable" WHERE app_group_id=%s', (group_id,)),
        lambda values: len(values) == 1 and _spend(values[0]) >= expected,
        seconds=70,
    )
    assert _spend(rows[0]) == pytest.approx(expected)


@pytest.mark.parametrize("duration", (None, "1mo"))
def test_app_group_customer_and_app_caps_apply_independently_across_workers(
    app_gateway: Gateway,
    duration: str | None,
) -> None:
    with wire_server(_reply) as wire, app_gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=f"{wire.url}/v1", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
        customer: Final = _customer(scenario)
        other: Final = _customer(scenario)
        group: Final = _group(scenario, {"max_budget": 10, "budget_duration": duration})
        chat, chat_key = _app(scenario, model, "chat")
        search, search_key = _app(scenario, model, "search")
        for app_id in (chat, search):
            bound: Final = app_gateway.request("PATCH", f"{_APPS}/{app_id}", {"app_group_id": group})
            assert bound.status_code == 200, bound.text
        child: Final = app_gateway.request(
            "PATCH", f"{_APPS}/{chat}", {"budget": {"max_budget": 0.05, "budget_duration": duration}}
        )
        assert child.status_code == 200, child.text
        pair: Final = app_gateway.request(
            "PUT",
            f"/management/v1/app_groups/{group}/customer_organizations/{customer.organization}",
            {"budget": {"max_budget": 0.11, "budget_duration": duration}},
        )
        assert pair.status_code == 200, pair.text
        assert app_gateway.request("GET", "/management/v1/app_groups", key=chat_key).status_code in (401, 403)
        assert _chat(app_gateway, chat_key, model, customer).status_code == 200
        _group_spend(app_gateway, group, _COST)
        _blocked_on_every_worker(app_gateway, chat_key, model, customer)
        assert _chat(app_gateway, search_key, model, customer).status_code == 200
        _group_spend(app_gateway, group, 2 * _COST)
        _blocked_on_every_worker(app_gateway, search_key, model, customer)
        assert _chat(app_gateway, search_key, model, other).status_code == 200
        _group_spend(app_gateway, group, 3 * _COST)
        lowered: Final = app_gateway.request(
            "PATCH", f"/management/v1/app_groups/{group}", {"budget": {"max_budget": 0.17, "budget_duration": duration}}
        )
        assert lowered.status_code == 200, lowered.text
        updated: Final = AppGroup.model_validate_json(lowered.content)
        assert updated.budget is not None and updated.budget.max_budget == 0.17
        assert updated.budget_usage[0].spend == pytest.approx(3 * _COST)
        wire.drain()
        _blocked_on_every_worker(app_gateway, search_key, model, other)
        assert not tuple(request for request in wire.drain() if request.method == "POST")
        readback: Final = JSON_OBJECT.validate_json(
            app_gateway.request("GET", f"/management/v1/app_groups/{group}").content
        )
        assert float(str(readback["spend"])) == pytest.approx(3 * _COST)


def test_moving_an_app_preserves_historical_group_usage(app_gateway: Gateway) -> None:
    from datetime import datetime, timezone

    with wire_server(_reply) as wire, app_gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=f"{wire.url}/v1", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
        customer: Final = _customer(scenario)
        first: Final = _group(scenario, {"max_budget": 10})
        second: Final = _group(scenario, {"max_budget": 10})
        app_id, key = _app(scenario, model, "moving")
        for group in (first, second):
            bound: Final = app_gateway.request("PATCH", f"{_APPS}/{app_id}", {"app_group_id": group})
            assert bound.status_code == 200, bound.text
            assert _chat(app_gateway, key, model, customer).status_code == 200
            _group_spend(app_gateway, group, _COST)
        day: Final = datetime.now(timezone.utc).date().isoformat()
        for group in (first, second):
            response: Final = app_gateway.client.get(
                "/management/v1/customer_organizations/daily/activity/aggregated",
                headers={"Authorization": f"Bearer {app_gateway.key}"},
                params={
                    "customer_org_id": customer.organization,
                    "app_group_id": group,
                    "start_date": day,
                    "end_date": day,
                },
            )
            assert response.status_code == 200, response.text
            metadata: Final = JSON_OBJECT.validate_python(JSON_OBJECT.validate_json(response.content)["metadata"])
            assert float(str(metadata["total_spend"])) == pytest.approx(_COST)
        totals: Final = app_gateway.rows('SELECT spend FROM "LiteLLM_AppTable" WHERE app_id=%s', (app_id,))
        assert _spend(totals[0]) == pytest.approx(2 * _COST)
