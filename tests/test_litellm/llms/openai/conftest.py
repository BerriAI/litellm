"""Shared fixtures for OpenAI provider tests."""

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    """Load bundled model map so PR-only model flags are visible in CI."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url)
    )
    litellm.add_known_models(model_cost_map=litellm.model_cost)
