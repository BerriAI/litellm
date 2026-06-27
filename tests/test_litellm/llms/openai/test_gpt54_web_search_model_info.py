"""Verify gpt-5.4 models expose supports_web_search in model_info."""

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.utils import get_model_info


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url)
    )
    litellm.add_known_models(model_cost_map=litellm.model_cost)


class TestGPT54WebSearch:
    """gpt-5.4 variants must have supports_web_search=True."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.4",
            "gpt-5.4-2026-03-05",
            "azure/gpt-5.4",
            "azure/gpt-5.4-2026-03-05",
        ],
    )
    def test_supports_web_search(self, model: str):
        info = get_model_info(model)
        assert info is not None, f"get_model_info returned None for {model}"
        assert (
            info["supports_web_search"] is True
        ), f"{model} should have supports_web_search=True"
