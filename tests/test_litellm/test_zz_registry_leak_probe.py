import litellm

from litellm.litellm_core_utils.model_registry import REGISTRY_SET_NAMES


def test_no_registry_names_are_shadowed():
    assert not (REGISTRY_SET_NAMES & litellm.__dict__.keys())
    assert not ({"models_by_provider", "model_list", "model_list_set"} & litellm.__dict__.keys())


def test_cost_map_survived_the_suite():
    assert len(litellm.model_cost) > 2000
    assert "claude-sonnet-4-5-20250929" in litellm.model_cost


def test_registry_still_rebuilds():
    litellm.model_cost["zz-probe-model"] = {"litellm_provider": "anthropic", "mode": "chat"}
    try:
        litellm.add_known_models()
        assert "zz-probe-model" in litellm.anthropic_models
        assert "zz-probe-model" in litellm.models_by_provider["anthropic"]
    finally:
        del litellm.model_cost["zz-probe-model"]
        litellm.add_known_models()


def test_registry_agrees_with_cost_map():
    assert litellm.anthropic_models <= litellm.model_cost.keys()
    assert litellm.groq_models <= litellm.model_cost.keys()
