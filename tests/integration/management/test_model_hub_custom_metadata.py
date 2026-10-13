import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.process import graceful_stop_seconds, owned_proxy_process

LIFECYCLE_METADATA: Final = {"retirement_date": "2027-03-01", "ptu_max_tpm": 300000, "provisioned_ptu": 100}


def _model_group(gateway: Gateway, model_group: str) -> dict[str, JsonValue]:
    data: Final = gateway.get("/model_group/info", params={"model_group": model_group})["data"]
    assert isinstance(data, list) and len(data) == 1, data
    return object_value(data[0])


def _deployment(gateway: Gateway, scenario: Scenario, model_name: str, model_info: dict[str, JsonValue]) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": f"{gateway.upstream_url}/v1",
            },
            "model_info": model_info,
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def _custom_fields(row: dict[str, JsonValue]) -> tuple[JsonValue, JsonValue]:
    assert "custom_tags" in row and "custom_metadata" in row, row
    return row["custom_tags"], row["custom_metadata"]


def _public_row(gateway: Gateway, path: str, model_group: str) -> dict[str, JsonValue] | None:
    response: Final = gateway.client.get(path)
    assert response.status_code == 200, response.text
    body: Final = response.json()
    rows: Final = body if isinstance(body, list) else object_value(body)["data"]
    assert isinstance(rows, list), response.text
    return next((object_value(row) for row in rows if object_value(row).get("model_group") == model_group), None)


def test_model_group_info_serves_the_custom_tags_and_metadata_set_on_the_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model_info={"custom_tags": ["deprecated", "ptu"], "custom_metadata": LIFECYCLE_METADATA}
        )
        group: Final = _model_group(gateway, model)
        assert group.get("custom_tags") == ["deprecated", "ptu"], group
        assert group.get("custom_metadata") == LIFECYCLE_METADATA, group


def test_model_group_info_merges_custom_tags_and_metadata_across_the_groups_deployments(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model_name: Final = f"integration-{uuid.uuid4().hex}"
        _deployment(
            gateway,
            scenario,
            model_name,
            {"custom_tags": ["ptu", "eu"], "custom_metadata": {"provisioned_ptu": 100, "region": "westeurope"}},
        )
        _deployment(
            gateway,
            scenario,
            model_name,
            {
                "custom_tags": ["eu", "deprecated"],
                "custom_metadata": {"provisioned_ptu": 50, "retirement_date": "2027-03-01"},
            },
        )
        group: Final = _model_group(gateway, model_name)
        assert sorted(string_value(tag) for tag in group.get("custom_tags") or []) == ["deprecated", "eu", "ptu"], group
        metadata: Final = object_value(group.get("custom_metadata"))
        assert metadata.get("region") == "westeurope", metadata
        assert metadata.get("retirement_date") == "2027-03-01", metadata
        assert metadata.get("provisioned_ptu") in (100, 50), metadata


def test_model_group_info_ignores_custom_metadata_values_that_are_not_scalars(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model_info={
                "custom_tags": ["ok", 7, {"nested": True}],
                "custom_metadata": {"owner": "platform", "nested": {"a": 1}, "list": [1, 2]},
            }
        )
        group: Final = _model_group(gateway, model)
        assert group.get("custom_tags") == ["ok"], group
        assert group.get("custom_metadata") == {"owner": "platform"}, group


@pytest.mark.parametrize("path", ["/public/model_hub", "/public/v1/model_hub"])
def test_public_model_hub_serves_the_custom_tags_and_metadata_of_a_public_model_group(
    gateway: Gateway, path: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model_info={"custom_tags": ["deprecated", "ptu"], "custom_metadata": LIFECYCLE_METADATA}
        )
        gateway.post("/model_group/make_public", {"model_groups": [model]})
        row: Final = eventually(
            lambda: _public_row(gateway, path, model),
            lambda found: found is not None,
            seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
        )
        assert row is not None
        assert _custom_fields(row) == (["deprecated", "ptu"], LIFECYCLE_METADATA), row


def test_patching_a_model_replaces_the_custom_tags_and_metadata_served_for_its_group(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model_name: Final = f"integration-{uuid.uuid4().hex}"
        identity: Final = _deployment(
            gateway, scenario, model_name, {"custom_tags": ["ptu"], "custom_metadata": LIFECYCLE_METADATA}
        )
        patched: Final = gateway.request(
            "PATCH",
            f"/model/{identity}/update",
            {"model_info": {"custom_tags": ["retiring"], "custom_metadata": {"retirement_date": "2027-06-01"}}},
        )
        assert patched.status_code == 200, patched.text
        group: Final = eventually(
            lambda: _model_group(gateway, model_name),
            lambda found: found.get("custom_tags") == ["retiring"],
            seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
        )
        assert _custom_fields(group) == (["retiring"], {"retirement_date": "2027-06-01"}), group


@pytest.mark.parametrize(
    "model_info",
    [
        pytest.param({"description": "plain model"}, id="missing"),
        pytest.param({"custom_tags": [], "custom_metadata": {}}, id="empty"),
        pytest.param({"custom_tags": None, "custom_metadata": None}, id="null"),
        pytest.param({"custom_tags": "ptu", "custom_metadata": "retiring"}, id="wrong-shape"),
        pytest.param(
            {"custom_tags": [1, None, {"a": 1}], "custom_metadata": {"nested": {"a": 1}}}, id="no-valid-entry"
        ),
    ],
)
def test_model_group_info_serves_null_custom_fields_when_the_model_sets_none_usable(
    gateway: Gateway, model_info: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model_info=model_info)
        group: Final = _model_group(gateway, model)
        assert _custom_fields(group) == (None, None), group


def test_model_group_info_keeps_long_values_whole_and_serves_a_repeated_tag_once(gateway: Gateway) -> None:
    long_value: Final = "x" * 5000
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model_info={"custom_tags": ["ptu", "ptu", "eu", "ptu"], "custom_metadata": {"notes": long_value}}
        )
        group: Final = _model_group(gateway, model)
        assert _custom_fields(group) == (["ptu", "eu"], {"notes": long_value}), group


def test_custom_tags_and_metadata_are_never_forwarded_to_the_provider(gateway: Gateway) -> None:
    marker: Final = f"custom-fields-{uuid.uuid4().hex}"
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(
            model_info={"custom_tags": ["deprecated"], "custom_metadata": {"retirement_date": "2027-03-01"}}
        )
        reply: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        assert reply.status_code == 200, reply.text
        observed: Final = upstream.get("/__observations")
        assert observed.status_code == 200, observed.text
        bodies: Final = [
            str(object_value(value)["body"])
            for value in object_value(observed.json())["requests"] or []
            if marker in str(object_value(value)["body"])
        ]
        assert len(bodies) == 1, bodies
        assert "custom_tags" not in bodies[0] and "2027-03-01" not in bodies[0], bodies[0]


def _yaml_config(directory: Path, upstream_url: str, model_name: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    deployment: Final = {
        "model_name": model_name,
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": f"{upstream_url}/v1", "api_key": "yaml-key"},
        "model_info": {"custom_tags": ["deprecated", "ptu"], "custom_metadata": LIFECYCLE_METADATA},
    }
    path: Final = directory / "custom_fields.yaml"
    path.write_text(yaml.safe_dump({**config, "model_list": [*config["model_list"], deployment]}))
    return path


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_yaml_custom_fields_survive_a_killed_worker_on_every_worker(gateway: Gateway, tmp_path: Path) -> None:
    model_name: Final = f"integration-yaml-{uuid.uuid4().hex}"
    config: Final = _yaml_config(tmp_path, gateway.upstream_url, model_name)
    with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
        victim: Final = min(owned.worker_pids())
        owned.kill_worker(victim)
        with ThreadPoolExecutor(max_workers=8) as pool:
            groups: Final = tuple(pool.map(lambda _: _model_group(owned.gateway, model_name), range(24)))
        assert all(_custom_fields(group) == (["deprecated", "ptu"], LIFECYCLE_METADATA) for group in groups), groups
