"""Partial-prefix wildcards like ``databricks/system.ai.*`` must splice the literal
prefix onto provider-stripped cost-map ids, never onto the provider-prefixed key."""

from collections.abc import Iterator
from typing import Final

import pytest

from integration._support.client import Gateway, object_value, string_value

PATTERNS: Final = ("databricks/system.ai.*", "databricks/*")


def _registered_model_id(gateway: Gateway, pattern: str) -> str:
    created: Final = object_value(
        gateway.post(
            "/model/new",
            {
                "model_name": pattern,
                "litellm_params": {
                    "model": pattern,
                    "api_key": "integration-wildcard-key",
                    "api_base": "https://example.invalid",
                },
            },
        )
    )
    return string_value(object_value(created["model_info"])["id"])


@pytest.fixture
def wildcard_deployments(gateway: Gateway) -> Iterator[None]:
    with gateway.scenario() as scenario:
        for pattern in PATTERNS:
            scenario.cleanups.callback(scenario.delete_model, _registered_model_id(gateway, pattern))
        yield


def _listed_model_ids(gateway: Gateway) -> tuple[str, ...]:
    data: Final = gateway.get("/v1/models")["data"]
    assert isinstance(data, list)
    return tuple(string_value(object_value(entry)["id"]) for entry in data)


def test_partial_prefix_wildcard_expands_to_unity_catalog_names(gateway: Gateway, wildcard_deployments: None) -> None:
    prefixed: Final = tuple(
        model_id for model_id in _listed_model_ids(gateway) if model_id.startswith("databricks/system.ai.")
    )
    assert prefixed
    assert all("/" not in model_id.removeprefix("databricks/system.ai.") for model_id in prefixed)


def test_provider_wildcard_still_expands_to_cost_map_names(gateway: Gateway, wildcard_deployments: None) -> None:
    assert any(model_id.startswith("databricks/databricks-") for model_id in _listed_model_ids(gateway))


def test_model_info_keeps_provider_model_for_expanded_deployments(gateway: Gateway, wildcard_deployments: None) -> None:
    data: Final = gateway.get("/model/info")["data"]
    assert isinstance(data, list)
    expanded: Final = [
        row
        for row in (object_value(entry) for entry in data)
        if string_value(row["model_name"]).startswith("databricks/system.ai.")
    ]
    assert expanded, "model/info returned no expanded rows for databricks/system.ai.*"
    bad: Final = [
        row["model_name"]
        for row in expanded
        if "system.ai.databricks/" in string_value(object_value(row["litellm_params"])["model"])
    ]
    assert not bad, f"litellm_params.model carries the corrupted expanded name: {bad}"
