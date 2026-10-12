import uuid
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value, string_value

LIFECYCLE_METADATA: Final = {"retirement_date": "2027-03-01", "ptu_max_tpm": 300000, "provisioned_ptu": 100}


def _model_group(gateway: Gateway, model_group: str) -> dict[str, JsonValue]:
    data: Final = gateway.get("/model_group/info", params={"model_group": model_group})["data"]
    assert isinstance(data, list) and len(data) == 1, data
    return object_value(data[0])


def _deployment(gateway: Gateway, scenario: Scenario, model_name: str, model_info: dict[str, JsonValue]) -> None:
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
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))


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
