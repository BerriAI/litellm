from typing import Final

from litellm.proxy.model_recommender.catalog import external_candidates
from litellm.proxy.model_recommender.models import Candidate


def test_catalog_deduplicates_deployments_and_ignores_ineligible_rows() -> None:
    entry: Final = {
        "mode": "chat",
        "litellm_provider": "openai",
        "input_cost_per_token": 0.1,
        "output_cost_per_token": 0.2,
        "supports_vision": True,
        "max_input_tokens": 128000,
    }
    result: Final = external_candidates(
        {
            "openai/configured_high": entry,
            "openai/new": entry,
            "openai/missing-price": {"mode": "chat", "litellm_provider": "openai"},
            "openai/embedding": {**entry, "mode": "embedding"},
            "other/new": {**entry, "litellm_provider": "other"},
            "malformed": "invalid",
        },
        (Candidate(model_group="alias", underlying_models=("openai/configured",)),),
    )
    assert len(result) == 1
    assert result[0].model_group == "openai/new"
    assert result[0].on_gateway is False
    assert result[0].supports_vision
    assert result[0].max_input_tokens == 128000
    assert result[0].output_cost_per_token == 0.2


def test_external_pool_is_bounded_and_empty_catalog_is_safe() -> None:
    entry: Final = {"mode": "chat", "litellm_provider": "openai", "input_cost_per_token": 0, "output_cost_per_token": 0}
    assert len(external_candidates({f"model-{index}": entry for index in range(100)}, ())) == 40
    assert external_candidates({}, ()) == ()
