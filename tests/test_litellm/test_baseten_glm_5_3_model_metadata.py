import pytest

import litellm


@pytest.fixture
def local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def test_entry_advertises_only_what_the_baseten_path_accepts(local_model_cost_map):
    """The Baseten path rejects unsupported request parameters."""
    supported = litellm.get_supported_openai_params(model="zai-org/GLM-5.3", custom_llm_provider="baseten")
    assert supported is not None

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.utils.get_optional_params(
            model="zai-org/GLM-5.3",
            custom_llm_provider="baseten",
            parallel_tool_calls=True,
            reasoning_effort="high",
            drop_params=False,
        )
