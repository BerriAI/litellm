"""Partial wildcards like ``cloudflare/@cf/*`` list each callable id once on /v1/models and /model/info,
and a partial wildcard that matches no known id, like ``databricks/system.ai.*``, lists no guessed ids."""

from collections.abc import Iterator
from typing import Final

import pytest

from integration._support.client import Gateway, object_value, string_value

PATTERNS: Final = ("cloudflare/@cf/*", "databricks/system.ai.*", "databricks/*")


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


def _model_info_names(gateway: Gateway) -> tuple[str, ...]:
    data: Final = gateway.get("/model/info")["data"]
    assert isinstance(data, list)
    return tuple(string_value(object_value(entry)["model_name"]) for entry in data)


def test_partial_wildcard_lists_provider_ids_once_on_both_endpoints(
    gateway: Gateway, wildcard_deployments: None
) -> None:
    listed: Final = tuple(model_id for model_id in _listed_model_ids(gateway) if model_id.startswith("cloudflare/"))
    model_info: Final = tuple(name for name in _model_info_names(gateway) if name.startswith("cloudflare/"))

    assert listed
    assert all(model_id.startswith("cloudflare/@cf/") and model_id.count("@cf/") == 1 for model_id in listed)
    assert len(listed) == len(set(listed))
    assert sorted(model_info) == sorted(listed)


def test_partial_wildcard_matching_no_known_id_lists_no_guessed_ids(
    gateway: Gateway, wildcard_deployments: None
) -> None:
    assert [model_id for model_id in _listed_model_ids(gateway) if model_id.startswith("databricks/system.ai.")] == []
    assert [name for name in _model_info_names(gateway) if name.startswith("databricks/system.ai.")] == [
        "databricks/system.ai.*"
    ]


def test_provider_wildcard_still_expands_to_cost_map_names(gateway: Gateway, wildcard_deployments: None) -> None:
    assert any(model_id.startswith("databricks/databricks-") for model_id in _listed_model_ids(gateway))
