import pytest

import litellm


def test_deepseek_model_listing_with_a_provider_check_lists_the_cost_map_models_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

    listed = litellm.get_valid_models(custom_llm_provider="deepseek", check_provider_endpoint=True)

    assert listed
    assert sorted(listed) == sorted(litellm.models_by_provider["deepseek"])
