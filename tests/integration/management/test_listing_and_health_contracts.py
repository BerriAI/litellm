import uuid
from datetime import datetime, timezone
from typing import Final

from pydantic import JsonValue

from litellm.litellm_core_utils.duration_parser import get_next_standardized_reset_time
from tests.integration._support.client import Gateway, Scenario, delete_key_if_present, object_value, string_value
from tests.integration._support.database import read_rows


def _utc(text: JsonValue) -> datetime:
    parsed: Final = datetime.fromisoformat(string_value(text))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _model_id(model_name: str) -> str:
    rows: Final = read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s', (model_name,))
    assert len(rows) == 1, rows
    return string_value(rows[0]["model_id"])


def _data(gateway: Gateway, path: str, key: str, params: dict[str, str] | None = None) -> list[dict[str, JsonValue]]:
    response: Final = gateway.request("GET", path, key=key, params=params)
    assert response.status_code == 200, response.text
    data: Final = object_value(response.json())["data"]
    assert isinstance(data, list)
    return [object_value(entry) for entry in data]


def _wildcard_model(gateway: Gateway, scenario: Scenario, prefix: str) -> None:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": f"{prefix}/*",
            "litellm_params": {
                "model": "anthropic/*",
                "api_key": "integration-provider-key",
                "api_base": gateway.upstream_url,
            },
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))


def test_budget_duration_schedules_reset_at_the_next_standardized_boundary(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=10.0, budget_duration="1d")
        rows: Final = read_rows(
            'SELECT created_at::text AS created_at, budget_reset_at::text AS reset_at FROM "LiteLLM_BudgetTable" '
            "WHERE budget_id = %s",
            (budget_id,),
        )
        assert len(rows) == 1, rows
        assert rows[0]["reset_at"] is not None, rows
        expected: Final = get_next_standardized_reset_time("1d", _utc(rows[0]["created_at"]), "UTC")
        assert abs((_utc(rows[0]["reset_at"]) - expected).total_seconds()) <= 3, (rows, expected)


def test_admin_health_counts_every_deployment_and_reports_a_live_one_healthy(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        model_id: Final = _model_id(model)
        report: Final = gateway.get("/health", {"model": model})
        assert (report["healthy_count"], report["unhealthy_count"]) == (1, 0), report
        healthy: Final = report["healthy_endpoints"]
        assert isinstance(healthy, list) and len(healthy) == 1, report
        assert object_value(healthy[0]).get("model_id") == model_id, report


def test_routes_listing_is_served_without_credentials(gateway: Gateway) -> None:
    response: Final = gateway.client.get("/routes")
    assert response.status_code == 200, response.text
    routes: Final = object_value(response.json())["routes"]
    assert isinstance(routes, list)
    assert {"/routes", "/key/generate"} <= {object_value(route)["path"] for route in routes}


def test_unrestricted_key_lists_models_and_none_when_only_access_groups_are_requested(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        grouped: Final = scenario.model(model_info={"access_groups": [f"integration-{uuid.uuid4().hex}"]})
        plain: Final = scenario.model()
        key: Final = scenario.key()
        listed: Final = {entry["id"] for entry in _data(gateway, "/models", key)}
        assert {grouped, plain} <= listed
        assert _data(gateway, "/models", key, {"only_model_access_groups": "True"}) == []


def test_model_info_by_id_matches_the_entry_in_the_keys_listing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        scenario.model()
        model_id: Final = _model_id(model)
        key: Final = scenario.key(models=[model])
        listing: Final = _data(gateway, "/model/info", key)
        assert {entry["model_name"] for entry in listing} == {model}
        listed: Final = [entry for entry in listing if object_value(entry["model_info"])["id"] == model_id]
        assert len(listed) == 1
        by_id: Final = _data(gateway, "/model/info", key, {"litellm_model_id": model_id})
        assert by_id == listed
        admin_by_id: Final = _data(gateway, "/model/info", gateway.key, {"litellm_model_id": model_id})
        assert [object_value(entry["model_info"])["id"] for entry in admin_by_id] == [model_id]


def test_model_group_info_for_a_personal_user_key_lists_only_the_users_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        scenario.model()
        created: Final = gateway.post("/user/new", {"user_id": f"integration-{uuid.uuid4().hex}", "models": [model]})
        scenario.cleanups.callback(scenario.delete_user, string_value(created["user_id"]))
        key: Final = string_value(created["key"])
        scenario.cleanups.callback(delete_key_if_present, gateway, key)
        groups: Final = [entry["model_group"] for entry in _data(gateway, "/model_group/info", key)]
        assert groups == [model]


def test_model_group_info_expands_a_wildcard_deployment_into_concrete_groups(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        prefix: Final = f"integration{uuid.uuid4().hex}"
        _wildcard_model(gateway, scenario, prefix)
        groups: Final = {
            string_value(entry["model_group"]) for entry in _data(gateway, "/model_group/info", gateway.key)
        }
        assert f"{prefix}/*" not in groups
        assert any(group.startswith(f"{prefix}/claude") for group in groups), sorted(groups)[:20]


def test_azure_deployment_route_denies_a_model_outside_the_keys_list(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        body: Final[dict[str, JsonValue]] = {"messages": [{"role": "user", "content": "integration control"}]}
        served: Final = gateway.request("POST", f"/openai/deployments/{allowed}/chat/completions", body, key=key)
        assert served.status_code == 200, served.text
        denied: Final = gateway.request("POST", f"/openai/deployments/{other}/chat/completions", body, key=key)
        assert denied.status_code == 403, denied.text
        assert "is not available for this API key" in denied.text
