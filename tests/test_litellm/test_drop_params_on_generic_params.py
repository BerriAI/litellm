"""Regression: drop_params must be a declared GenericLiteLLMParams field."""

from litellm.types.router import GenericLiteLLMParams


def test_generic_litellm_params_declares_drop_params() -> None:
    assert "drop_params" in GenericLiteLLMParams.model_fields
    params = GenericLiteLLMParams(drop_params=True)
    assert params.drop_params is True
    dumped = params.model_dump(exclude_none=True)
    assert dumped.get("drop_params") is True


def test_partial_generic_params_default_drop_params_to_none() -> None:
    patch = GenericLiteLLMParams(api_base="https://example.com")
    dumped = patch.model_dump()
    assert "drop_params" in dumped
    assert dumped["drop_params"] is None
