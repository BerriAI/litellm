import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.openai.openai import OpenAIConfig


@pytest.fixture()
def config() -> OpenAIConfig:
    return OpenAIConfig()


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))
    litellm.add_known_models(model_cost_map=litellm.model_cost)


def test_gpt5_rejects_reasoning_effort_none_for_unsupported_models(config: OpenAIConfig):
    """Models marked supports_none_reasoning_effort=false must not forward reasoning_effort=none."""
    for model in ("gpt-5", "gpt-5-mini"):
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"reasoning_effort": "none"},
                optional_params={},
                model=model,
                drop_params=False,
            )

        # Dict form {"effort": "none"} should be gated the same way.
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"reasoning_effort": {"effort": "none", "summary": "detailed"}},
                optional_params={},
                model=model,
                drop_params=False,
            )


def test_gpt5_drops_reasoning_effort_none_when_requested(config: OpenAIConfig):
    """drop_params=True should strip unsupported reasoning_effort=none instead of raising."""
    for model in ("gpt-5", "gpt-5-mini"):
        params = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model=model,
            drop_params=True,
        )
        assert "reasoning_effort" not in params


def test_gpt5_1_passes_through_reasoning_effort_none(config: OpenAIConfig):
    """Models marked supports_none_reasoning_effort=true still forward reasoning_effort=none."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="gpt-5.1",
        drop_params=False,
    )
    assert params["reasoning_effort"] == "none"
