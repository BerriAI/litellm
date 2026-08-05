import pytest

import litellm
from litellm import Router

BACKEND_MODEL = "deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731"
PUBLIC_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
INPUT_COST = 7.2e-08
OUTPUT_COST = 1.44e-07
CACHE_READ_COST = 1.44e-08


def _pricing_model_info():
    return {
        "input_cost_per_token": INPUT_COST,
        "output_cost_per_token": OUTPUT_COST,
        "cache_read_input_token_cost": CACHE_READ_COST,
    }


def _nested_config():
    """The reporter's config shape: model_info nested inside litellm_params."""
    return {
        "model_name": PUBLIC_MODEL,
        "litellm_params": {
            "model": BACKEND_MODEL,
            "api_key": "fake-key",
            "extra_body": {"service_tier": "flex"},
            "model_info": _pricing_model_info(),
        },
    }


def _deployment_level_config():
    return {
        "model_name": PUBLIC_MODEL,
        "litellm_params": {
            "model": BACKEND_MODEL,
            "api_key": "fake-key",
            "extra_body": {"service_tier": "flex"},
        },
        "model_info": _pricing_model_info(),
    }


@pytest.fixture(autouse=True)
def clean_cost_map():
    """Keep registrations from bleeding across tests."""
    from litellm.utils import _invalidate_model_cost_lowercase_map

    def _clear():
        litellm.model_cost.pop(BACKEND_MODEL, None)
        litellm.model_cost.pop(PUBLIC_MODEL, None)
        _invalidate_model_cost_lowercase_map()

    _clear()
    yield
    _clear()


def _deployment_pricing(router):
    deployment = router.model_list[0]
    return deployment.get("model_info", {})


def test_nested_model_info_pricing_is_applied():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/35691

    `model_info` is a declared field on GenericLiteLLMParams, so nesting it under
    `litellm_params` validates cleanly and is then silently ignored by every cost
    path, leaving the deployment with no pricing at all.
    """
    router = Router(model_list=[_nested_config()])
    model_info = _deployment_pricing(router)

    assert model_info.get("input_cost_per_token") == INPUT_COST
    assert model_info.get("output_cost_per_token") == OUTPUT_COST
    assert model_info.get("cache_read_input_token_cost") == CACHE_READ_COST


def test_nested_matches_deployment_level():
    """Both spellings must end up with the same pricing."""
    nested = _deployment_pricing(Router(model_list=[_nested_config()]))
    proper = _deployment_pricing(Router(model_list=[_deployment_level_config()]))

    for field in (
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
    ):
        assert nested.get(field) == proper.get(field)


def test_deployment_level_model_info_wins_over_nested():
    """An explicit deployment-level value is not overwritten by a nested one."""
    config = _deployment_level_config()
    config["litellm_params"]["model_info"] = {"input_cost_per_token": 999.0}

    model_info = _deployment_pricing(Router(model_list=[config]))
    assert model_info.get("input_cost_per_token") == INPUT_COST


def test_nested_model_info_does_not_override_deployment_id():
    """The generated deployment id must survive a nested `id`."""
    config = _nested_config()
    config["litellm_params"]["model_info"]["id"] = "nested-id-should-be-ignored"

    model_info = _deployment_pricing(Router(model_list=[config]))
    assert model_info.get("id") != "nested-id-should-be-ignored"


def test_no_nested_model_info_is_unchanged():
    """Configs without the nested block keep working exactly as before."""
    model_info = _deployment_pricing(Router(model_list=[_deployment_level_config()]))
    assert model_info.get("input_cost_per_token") == INPUT_COST


@pytest.mark.asyncio
async def test_nested_model_info_produces_non_zero_response_cost():
    """
    End to end: the persisted spend is derived from `response_cost`, so pricing that
    never reaches the cost map shows up as spend 0 in LiteLLM_SpendLogs.
    """
    router = Router(model_list=[_nested_config()])

    response = await router.acompletion(
        model=PUBLIC_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )

    response_cost = response._hidden_params.get("response_cost")
    assert response_cost is not None
    assert response_cost > 0
