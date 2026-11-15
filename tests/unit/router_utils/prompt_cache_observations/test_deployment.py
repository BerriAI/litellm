from collections.abc import Iterator, Mapping

import pytest

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.model_cost_table import published
from litellm.router_utils.prompt_cache_observations.deployment import (
    SelectedDeployment,
    deployment_cache_identity,
    effective_destination,
    exact_min_prefix_tokens,
    selected_deployment,
)

_KEY = b"deployment-identity-test-key"
_SELECTED = SelectedDeployment(deployment_id="deployment-1", base_model=None)
_BEDROCK_EAST = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-test-v1:0"


def _fingerprint(
    destination: str | None = _BEDROCK_EAST,
    model: str = "us.anthropic.claude-test-v1:0",
    selected: SelectedDeployment = _SELECTED,
) -> str:
    return deployment_cache_identity(
        selected=selected,
        provider="bedrock",
        provider_model=model,
        destination=destination,
        fingerprint_key=_KEY,
        min_prefix_tokens=lambda _model, _provider: 1024,
    ).config_fingerprint


def test_streaming_and_non_streaming_calls_to_one_destination_share_a_fingerprint() -> None:
    assert effective_destination(f"{_BEDROCK_EAST}/converse") == effective_destination(
        f"{_BEDROCK_EAST}/converse-stream"
    )
    assert _fingerprint(effective_destination(f"{_BEDROCK_EAST}/invoke")) == _fingerprint(
        effective_destination(f"{_BEDROCK_EAST}/invoke-with-response-stream")
    )


def test_vertex_stream_and_non_stream_operations_share_a_destination() -> None:
    base = "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5/publishers/anthropic/models/claude-test"
    assert effective_destination(f"{base}:rawPredict") == effective_destination(f"{base}:streamRawPredict") == base


def test_destination_drops_the_query_string() -> None:
    assert effective_destination("https://example.services.ai.azure.com/anthropic/v1/messages?api-key=secret") == (
        "https://example.services.ai.azure.com/anthropic/v1/messages"
    )


@pytest.mark.parametrize(
    ("destination", "model", "selected"),
    [
        ("https://bedrock-runtime.eu-west-1.amazonaws.com/model/us.anthropic.claude-test-v1:0", None, _SELECTED),
        (_BEDROCK_EAST, "us.anthropic.claude-test-v2:0", _SELECTED),
        (_BEDROCK_EAST, None, SelectedDeployment(deployment_id="deployment-1", base_model="claude-test")),
    ],
)
def test_region_resolved_at_request_time_model_and_base_model_change_the_fingerprint(
    destination: str, model: str | None, selected: SelectedDeployment
) -> None:
    changed = _fingerprint(destination, model or "us.anthropic.claude-test-v1:0", selected)
    assert changed != _fingerprint()


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        ({"id": "d1", "prompt_cache_observation": True}, SelectedDeployment(deployment_id="d1", base_model=None)),
        (
            {"id": "d1", "prompt_cache_observation": True, "base_model": "claude-test", "rpm": 10},
            SelectedDeployment(deployment_id="d1", base_model="claude-test"),
        ),
        ({"id": "d1"}, None),
        ({"id": "d1", "prompt_cache_observation": False}, None),
        ({"prompt_cache_observation": True}, None),
        (None, None),
    ],
)
def test_only_deployments_that_opt_in_are_observed(model_info: object, expected: SelectedDeployment | None) -> None:
    assert selected_deployment(model_info) == expected


def _rows_of(provider: str) -> Iterator[tuple[str, Mapping[str, object]]]:
    for name, row in published().items():
        info = as_str_mapping(row)
        if info is not None and info.get("litellm_provider") == provider:
            yield name, info


def _model_with_minimum(provider: str, prefix: str = "") -> tuple[str, object]:
    return next(
        (name.removeprefix(prefix), info.get("prompt_cache_min_tokens"))
        for name, info in _rows_of(provider)
        if isinstance(info.get("prompt_cache_min_tokens"), int) and name.startswith(prefix)
    )


@pytest.mark.parametrize(
    ("provider", "cost_map_provider", "prefix"),
    [
        ("anthropic", "anthropic", ""),
        ("azure_ai", "azure_ai", "azure_ai/"),
        ("vertex_ai", "vertex_ai-anthropic_models", "vertex_ai/"),
        ("bedrock", "bedrock_converse", ""),
    ],
)
def test_exact_minimum_is_the_cost_map_row_for_the_served_model(
    provider: str, cost_map_provider: str, prefix: str
) -> None:
    model, minimum = _model_with_minimum(cost_map_provider, prefix)
    assert exact_min_prefix_tokens(model, provider) == minimum


def test_bedrock_route_prefix_names_the_same_model() -> None:
    model, minimum = _model_with_minimum("bedrock_converse")
    assert exact_min_prefix_tokens(f"converse/{model}", "bedrock") == minimum
    assert exact_min_prefix_tokens(f"invoke/{model}", "bedrock") == minimum


@pytest.mark.parametrize("provider", ["azure_ai", "vertex_ai", "bedrock"])
def test_a_row_for_another_provider_never_supplies_the_minimum(provider: str) -> None:
    table = published()
    model = next(
        name
        for name, info in _rows_of("anthropic")
        if isinstance(info.get("prompt_cache_min_tokens"), int) and f"{provider}/{name}" not in table
    )
    assert exact_min_prefix_tokens(model, provider) is None


def test_unlisted_or_upgraded_model_has_no_minimum_instead_of_a_neighbours() -> None:
    model, _ = _model_with_minimum("anthropic")
    assert exact_min_prefix_tokens(f"{model}-next", "anthropic") is None
    assert exact_min_prefix_tokens("claude-model-that-is-not-in-the-cost-map", "anthropic") is None
