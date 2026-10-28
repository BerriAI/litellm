"""
Regression test for Issue #47675:
Verifies that get_model_cost_information falls back to bare model name lookup
when deployment custom_llm_provider does not match the provider key in model pricing map.
"""
import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.types.utils import ModelResponse


def test_cross_key_provider_mismatch_fallback():
    """
    Test that when custom_llm_provider='anthropic' is used with a model priced under
    another provider (e.g., 'tencent/deepseek-v4.1-flash'), the fallback populates
    model_map_value instead of silently returning None ($0 spend).
    """
    setup = StandardLoggingPayloadSetup()
    model = "tencent/deepseek-v4.1-flash"
    mock_response = ModelResponse(model=model)

    info = setup.get_model_cost_information(
        base_model=model,
        custom_pricing=False,
        custom_llm_provider="anthropic",
        init_response_obj=mock_response,
    )

    assert info.get("model_map_key") == model
    assert info.get("model_map_value") is not None
    assert info["model_map_value"].get("litellm_provider") == "tencent"
    assert info["model_map_value"].get("input_cost_per_token") == 3e-07


def test_unmapped_model_safely_returns_none():
    """
    Test that an unknown model that does not exist in the pricing map
    safely returns model_map_value=None without raising unhandled exceptions.
    """
    setup = StandardLoggingPayloadSetup()
    fake_model = "fake_provider/completely_non_existent_model_xyz"
    mock_response = ModelResponse(model=fake_model)

    info = setup.get_model_cost_information(
        base_model=fake_model,
        custom_pricing=False,
        custom_llm_provider="anthropic",
        init_response_obj=mock_response,
    )

    assert info.get("model_map_key") == fake_model
    assert info.get("model_map_value") is None
