from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.model_discovery import discover_model_metadata
from litellm.router_utils.model_request_defaults import deployment_params_with_request_defaults, model_request_defaults
from litellm.types.proxy.model_metadata import GatewayModelMetadata, ModelRequestDefaults

_POLICY: Final = {
    "output_token_budget": 8192,
    "output_token_budget_by_reasoning_effort": {"low": 65536, "high": 65536, "xhigh": 65536, "max": 131072},
}


@pytest.mark.parametrize(
    "effort,expected", [(None, 8192), ("off", 8192), ("low", 65536), ("high", 65536), ("xhigh", 65536), ("max", 131072)]
)
def test_request_budget_follows_effective_reasoning_without_claiming_supplier_limit(effort, expected):
    metadata: Final = {"context_window": 262144, "request_defaults": _POLICY}
    request: Final = {"reasoning_effort": effort}
    assert model_request_defaults(metadata, {}, request, "completion") == {"max_tokens": expected}
    assert "max_output_tokens" not in discover_model_metadata((GatewayModelMetadata.model_validate(metadata),))
    assert request == {"reasoning_effort": effort}


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
def test_explicit_caller_budget_wins(key):
    assert (
        model_request_defaults({"request_defaults": _POLICY}, {}, {key: 123, "reasoning_effort": "max"}, "completion")
        == {}
    )


@pytest.mark.parametrize(
    "surface,key",
    [
        ("completion", "max_tokens"),
        ("acompletion", "max_tokens"),
        ("responses", "max_output_tokens"),
        ("aresponses", "max_output_tokens"),
        ("anthropic_messages", "max_tokens"),
    ],
)
def test_budget_uses_endpoint_parameter_and_request_reasoning_overrides_deployment(surface, key):
    assert model_request_defaults(
        {"request_defaults": _POLICY},
        {"reasoning_effort": "max", "max_tokens": 456},
        {"reasoning": {"effort": "low"}},
        surface,
    ) == {key: 65536}


def test_deployment_and_metadata_default_efforts_are_resolved():
    metadata: Final = {"request_defaults": _POLICY, "default_reasoning_effort": "high"}
    assert model_request_defaults(metadata, {}, {}, "completion") == {"max_tokens": 65536}
    assert model_request_defaults(metadata, {"reasoning": {"effort": "max"}}, {}, "responses") == {
        "max_output_tokens": 131072
    }
    assert model_request_defaults(metadata, {}, {}, "aembedding") == {}
    assert model_request_defaults({}, {}, {}, "completion") == {}
    assert model_request_defaults(metadata, {}, {"reasoning_effort": 50}, "completion") == {"max_tokens": 8192}


def test_selected_policy_replaces_competing_static_budget_aliases():
    deployment: Final = {"model": "openai/fixture", "max_tokens": 1, "max_completion_tokens": 2, "max_output_tokens": 3}
    defaults: Final = model_request_defaults(
        {"request_defaults": _POLICY}, deployment, {"reasoning_effort": "max"}, "responses"
    )
    params: Final = deployment_params_with_request_defaults(deployment, defaults)
    assert {**params, **defaults} == {"model": "openai/fixture", "max_output_tokens": 131072}
    assert deployment_params_with_request_defaults(deployment, {}) == deployment
    assert deployment_params_with_request_defaults(
        deployment, {}, {"request_defaults": _POLICY}, {"max_tokens": 123}
    ) == {"model": "openai/fixture"}
    assert deployment["max_completion_tokens"] == 2


@pytest.mark.parametrize("value", [0, -1, True, "8192", 1.5])
@pytest.mark.parametrize("field", ["output_token_budget", "output_token_budget_by_reasoning_effort"])
def test_invalid_request_budgets_are_rejected(value, field):
    invalid: Final = {field: {"high": value} if field.endswith("effort") else value}
    with pytest.raises(ValidationError):
        ModelRequestDefaults.model_validate(invalid)


def test_request_budget_maps_are_immutable_and_only_common_policy_is_advertised():
    metadata: Final = GatewayModelMetadata.model_validate({"request_defaults": _POLICY})
    assert discover_model_metadata((metadata, metadata)) == {"request_defaults": _POLICY}
    assert discover_model_metadata((metadata, GatewayModelMetadata())) == {}
    other: Final = GatewayModelMetadata(request_defaults=ModelRequestDefaults(output_token_budget=16384))
    assert discover_model_metadata((metadata, other)) == {}
    defaults: Final = metadata.request_defaults
    assert defaults is not None
    with pytest.raises(TypeError):
        defaults.output_token_budget_by_reasoning_effort["max"] = 1


@pytest.mark.parametrize("surface,key", [("acompletion", "max_tokens"), ("aresponses", "max_output_tokens")])
def test_structured_reasoning_effort_preserves_summary_and_caller_precedence(surface, key):
    metadata: Final = {"request_defaults": _POLICY}
    deployment: Final = {"reasoning_effort": {"effort": "max", "summary": "auto"}}
    request: Final = {"reasoning_effort": {"effort": "high", "summary": "auto"}}
    assert model_request_defaults(metadata, deployment, request, surface) == {key: 65536}
    assert model_request_defaults(metadata, deployment, {}, surface) == {key: 131072}
    assert request == {"reasoning_effort": {"effort": "high", "summary": "auto"}}
    assert deployment == {"reasoning_effort": {"effort": "max", "summary": "auto"}}
