"""
Tests for the Volcengine Ark model catalog entries.

These assert the properties that were checked against the live Ark API
(POST https://ark.cn-beijing.volces.com/api/v3/chat/completions) so a future
edit cannot silently contradict them.
"""

import pytest

from litellm import get_model_info


SEED_2_1_FLAGSHIPS = [
    "volcengine/doubao-seed-2-1-pro-260628",
    "volcengine/doubao-seed-2-1-turbo-260628",
]


@pytest.mark.parametrize("model", SEED_2_1_FLAGSHIPS)
def test_seed_2_1_flagship_is_registered(model):
    """Both flagships must resolve through get_model_info with the volcengine provider."""
    info = get_model_info(model=model)
    assert info["litellm_provider"] == "volcengine"
    assert info["mode"] == "chat"


@pytest.mark.parametrize("model", SEED_2_1_FLAGSHIPS)
def test_seed_2_1_flagship_limits(model):
    """Ark documents 256k context and 256k max output for both models.

    The 256k output ceiling was also probed directly: max_tokens=256000 is
    accepted and 300000 is rejected with "max_tokens ... not valid".
    """
    info = get_model_info(model=model)
    assert info["max_input_tokens"] == 256000
    assert info["max_output_tokens"] == 256000


@pytest.mark.parametrize("model", SEED_2_1_FLAGSHIPS)
def test_seed_2_1_flagship_capabilities(model):
    """Reasoning, vision and function calling were each confirmed against the API.

    supports_tool_choice stays False, matching the other volcengine entries:
    a forced function is honoured, but tool_choice="none" is ignored and the
    model calls the tool anyway, so the parameter is not fully supported.
    """
    info = get_model_info(model=model)
    assert info["supports_reasoning"] is True
    assert info["supports_vision"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is False


def test_seed_2_1_pro_is_priced_above_turbo():
    """Ark lists pro at CNY 6/30 per 1M and turbo at CNY 3/15, i.e. turbo is half.

    Asserting the ratio rather than absolute figures keeps the test meaningful
    if the CNY-to-USD rate is refreshed later.
    """
    pro = get_model_info(model="volcengine/doubao-seed-2-1-pro-260628")
    turbo = get_model_info(model="volcengine/doubao-seed-2-1-turbo-260628")

    assert pro["input_cost_per_token"] == pytest.approx(
        turbo["input_cost_per_token"] * 2, rel=1e-6
    )
    assert pro["output_cost_per_token"] == pytest.approx(
        turbo["output_cost_per_token"] * 2, rel=1e-6
    )
    # cache-hit pricing is a fifth of the input price on both
    for info in (pro, turbo):
        assert info["cache_read_input_token_cost"] == pytest.approx(
            info["input_cost_per_token"] / 5, rel=1e-6
        )
