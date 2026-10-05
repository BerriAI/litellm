from typing import Final

import pytest

from litellm.proxy.model_discovery import discover_model_metadata, guaranteed_token_limit
from litellm.types.proxy.model_metadata import GatewayModelMetadata, resolve_gateway_model_metadata


def test_configured_overrides_preserve_false_empty_and_independent_limits():
    metadata: Final = resolve_gateway_model_metadata(
        {
            "context_window": 1200,
            "max_input_tokens": 1100,
            "max_output_tokens": 200,
            "supports_function_calling": True,
            "reasoning_effort_levels": ["low", "max"],
            "default_reasoning_effort": "low",
        },
        {"context_window": 1000, "supports_function_calling": False, "reasoning_effort_levels": []},
    )
    assert metadata.context_window == 1000
    assert metadata.max_input_tokens == 1100
    assert metadata.max_output_tokens == 200
    assert discover_model_metadata((metadata,)) == {
        "context_window": 1000,
        "supports_function_calling": False,
        "reasoning_effort_levels": [],
    }


def test_mixed_deployments_advertise_only_guaranteed_metadata():
    sources: Final = (
        GatewayModelMetadata(
            context_window=1200,
            max_input_tokens=1100,
            max_output_tokens=200,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
            reasoning_effort_levels=["low", "high", "max"],
            default_reasoning_effort="high",
            supported_modalities=["text", "image"],
        ),
        GatewayModelMetadata(
            context_window=800,
            max_input_tokens=700,
            max_output_tokens=100,
            supports_function_calling=False,
            reasoning_effort_levels=["low", "high"],
            default_reasoning_effort="low",
            supported_modalities=["text"],
        ),
    )
    assert discover_model_metadata(sources) == {
        "context_window": 800,
        "supports_function_calling": False,
        "reasoning_effort_levels": ["low", "high"],
        "supported_modalities": ["text"],
    }
    assert guaranteed_token_limit(sources, "max_input_tokens") == 700
    assert guaranteed_token_limit(sources, "max_output_tokens") == 100


@pytest.mark.parametrize("value", [None, True, 0, -1, "1200", 1.5])
def test_unknown_or_invalid_context_never_fabricates_capacity(value):
    metadata: Final = GatewayModelMetadata(context_window=value, max_input_tokens=1100, max_output_tokens=200)
    assert discover_model_metadata((metadata,)) == {}
    assert discover_model_metadata((GatewayModelMetadata(context_window=1200), metadata)) == {}
