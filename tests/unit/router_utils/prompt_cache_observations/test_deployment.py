from collections.abc import Iterator, Mapping

import pytest
from pydantic import JsonValue

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.model_cost_table import published

from litellm.router_utils.prompt_cache_observations.deployment import (
    deployment_cache_identity,
    exact_min_prefix_tokens,
)
from litellm.router_utils.prompt_cache_observations.types import DeploymentCacheIdentity

_KEY = b"deployment-identity-test-key"
_BASE_PARAMS: dict[str, JsonValue] = {
    "model": "bedrock/us.anthropic.claude-test-v1:0",
    "aws_region_name": "us-east-1",
    "api_key": "first-key",
    "rpm": 10,
    "timeout": 30,
}


def _identity(params: Mapping[str, JsonValue], model: str = "us.anthropic.claude-test-v1:0") -> DeploymentCacheIdentity:
    return deployment_cache_identity(
        deployment_id="deployment-1",
        provider="bedrock",
        provider_model=model,
        base_model=None,
        litellm_params=params,
        fingerprint_key=_KEY,
        min_prefix_tokens=lambda _model, _provider: 1024,
    )


def test_config_changes_unrelated_to_caching_keep_the_same_fingerprint() -> None:
    changed: dict[str, JsonValue] = {
        **_BASE_PARAMS,
        "api_key": "rotated-key",
        "rpm": 99,
        "timeout": 5,
        "tags": ["blue"],
    }
    assert _identity(changed) == _identity(_BASE_PARAMS)


@pytest.mark.parametrize(
    ("params", "model"),
    [
        ({**_BASE_PARAMS, "aws_region_name": "eu-west-1"}, "us.anthropic.claude-test-v1:0"),
        (_BASE_PARAMS, "us.anthropic.claude-test-v2:0"),
        ({**_BASE_PARAMS, "api_base": "https://bedrock-runtime.example"}, "us.anthropic.claude-test-v1:0"),
        (
            {**_BASE_PARAMS, "cache_control_injection_points": [{"location": "message", "role": "system"}]},
            "us.anthropic.claude-test-v1:0",
        ),
    ],
)
def test_cache_relevant_config_changes_change_the_fingerprint(params: Mapping[str, JsonValue], model: str) -> None:
    assert _identity(params, model).config_fingerprint != _identity(_BASE_PARAMS).config_fingerprint


def test_gateway_injection_is_recorded_from_the_deployment_config() -> None:
    injected: dict[str, JsonValue] = {
        **_BASE_PARAMS,
        "cache_control_injection_points": [{"location": "message", "role": "system"}],
    }
    assert _identity(injected).gateway_injects_breakpoints is True
    assert _identity(_BASE_PARAMS).gateway_injects_breakpoints is False


def test_minimum_is_looked_up_for_the_base_model_when_one_is_configured() -> None:
    looked_up: list[tuple[str, str]] = []

    def minimum(model: str, provider: str) -> int | None:
        looked_up.append((model, provider))
        return 2048

    identity = deployment_cache_identity(
        deployment_id="deployment-1",
        provider="azure_ai",
        provider_model="my-claude-deployment",
        base_model="claude-test",
        litellm_params={"model": "azure_ai/my-claude-deployment"},
        fingerprint_key=_KEY,
        min_prefix_tokens=minimum,
    )
    assert (identity.min_prefix_tokens, looked_up) == (2048, [("claude-test", "azure_ai")])


def _anthropic_cost_rows() -> Iterator[tuple[str, Mapping[str, object]]]:
    for name, row in published().items():
        info = as_str_mapping(row)
        if info is not None and info.get("litellm_provider") == "anthropic":
            yield name, info


def test_exact_minimum_matches_the_cost_map_entry_for_the_model() -> None:
    model, minimum = next(
        (name, info.get("prompt_cache_min_tokens"))
        for name, info in _anthropic_cost_rows()
        if isinstance(info.get("prompt_cache_min_tokens"), int)
    )
    assert exact_min_prefix_tokens(model, "anthropic") == minimum


def test_unknown_model_has_no_minimum_instead_of_the_generic_default() -> None:
    assert exact_min_prefix_tokens("claude-model-that-is-not-in-the-cost-map", "anthropic") is None
