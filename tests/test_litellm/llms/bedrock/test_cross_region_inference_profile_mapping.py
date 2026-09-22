"""Test Bedrock cross-region inference profile model mapping"""

from typing import Final, NamedTuple

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.common_utils import BedrockModelInfo
from litellm.types.utils import (
    Choices,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Resolve models against this checkout's cost map instead of the network-fetched
    ``main`` copy, which lags this branch until merge."""
    original_converse_models = set(litellm.bedrock_converse_models)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    try:
        litellm.bedrock_converse_models.update(
            key
            for key, value in litellm.model_cost.items()
            if isinstance(value, dict) and value.get("litellm_provider") == "bedrock_converse"
        )
        yield
    finally:
        litellm.bedrock_converse_models.clear()
        litellm.bedrock_converse_models.update(original_converse_models)
        litellm.get_model_info.cache_clear()


KIMI_K3_KEYS: Final = ("moonshotai.kimi-k3", "global.moonshotai.kimi-k3", "us.moonshotai.kimi-k3")
NON_PRICE_FIELDS: Final = frozenset(
    {
        "mode",
        "litellm_provider",
        "max_input_tokens",
        "max_output_tokens",
        "max_tokens",
        "supports_reasoning",
        "supports_function_calling",
        "supports_prompt_caching",
    }
)


class GptProfile(NamedTuple):
    model_id: str
    input_cost: float
    input_cost_above_272k: float
    cache_write: float
    cache_write_above_272k: float
    cache_read: float
    cache_read_above_272k: float
    output_cost: float
    output_cost_above_272k: float


GPT_5_6_PROFILES = [
    GptProfile(
        model_id="us.openai.gpt-5.6-sol",
        input_cost=4.4e-06,
        input_cost_above_272k=8.8e-06,
        cache_write=5.5e-06,
        cache_write_above_272k=1.1e-05,
        cache_read=4.4e-07,
        cache_read_above_272k=8.8e-07,
        output_cost=2.2e-05,
        output_cost_above_272k=3.3e-05,
    ),
    GptProfile(
        model_id="global.openai.gpt-5.6-sol",
        input_cost=4e-06,
        input_cost_above_272k=8e-06,
        cache_write=5e-06,
        cache_write_above_272k=1e-05,
        cache_read=4e-07,
        cache_read_above_272k=8e-07,
        output_cost=2e-05,
        output_cost_above_272k=3e-05,
    ),
    GptProfile(
        model_id="us.openai.gpt-5.6-terra",
        input_cost=2.2e-06,
        input_cost_above_272k=4.4e-06,
        cache_write=2.75e-06,
        cache_write_above_272k=5.5e-06,
        cache_read=2.2e-07,
        cache_read_above_272k=4.4e-07,
        output_cost=1.32e-05,
        output_cost_above_272k=1.98e-05,
    ),
    GptProfile(
        model_id="global.openai.gpt-5.6-terra",
        input_cost=2e-06,
        input_cost_above_272k=4e-06,
        cache_write=2.5e-06,
        cache_write_above_272k=5e-06,
        cache_read=2e-07,
        cache_read_above_272k=4e-07,
        output_cost=1.2e-05,
        output_cost_above_272k=1.8e-05,
    ),
    GptProfile(
        model_id="us.openai.gpt-5.6-luna",
        input_cost=2.2e-07,
        input_cost_above_272k=4.4e-07,
        cache_write=2.75e-07,
        cache_write_above_272k=5.5e-07,
        cache_read=2.2e-08,
        cache_read_above_272k=4.4e-08,
        output_cost=1.32e-06,
        output_cost_above_272k=1.98e-06,
    ),
    GptProfile(
        model_id="global.openai.gpt-5.6-luna",
        input_cost=2e-07,
        input_cost_above_272k=4e-07,
        cache_write=2.5e-07,
        cache_write_above_272k=5e-07,
        cache_read=2e-08,
        cache_read_above_272k=4e-08,
        output_cost=1.2e-06,
        output_cost_above_272k=1.8e-06,
    ),
]


def _bedrock_response(model, usage):
    return ModelResponse(
        id="test",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="OK", role="assistant"),
            )
        ],
        usage=usage,
    )


def test_kimi_k3_siblings_agree(local_model_cost_map: None) -> None:
    entries: Final = tuple(litellm.model_cost[k] for k in KIMI_K3_KEYS)
    for field in NON_PRICE_FIELDS:
        assert len({repr(e.get(field)) for e in entries}) == 1, field
    for key in KIMI_K3_KEYS:
        assert BedrockModelInfo.get_bedrock_route(f"bedrock/{key}") == "converse"


@pytest.mark.parametrize("key", KIMI_K3_KEYS)
def test_kimi_k3_key_resolves_to_itself_and_costs_rate_times_tokens(key: str, local_model_cost_map: None) -> None:
    info: Final = litellm.get_model_info(f"bedrock/{key}")
    assert info["key"] == key, "resolved through the CRIS-to-parent fallback instead of its own entry"
    r: Final = ModelResponse(model=key, usage=Usage(prompt_tokens=87, completion_tokens=133, total_tokens=220))
    expected: Final = 87 * info["input_cost_per_token"] + 133 * info["output_cost_per_token"]
    assert completion_cost(completion_response=r, custom_llm_provider="bedrock") == pytest.approx(expected)


@pytest.mark.parametrize("profile", GPT_5_6_PROFILES, ids=lambda p: p.model_id)
def test_bedrock_gpt_5_6_profiles_route_to_converse(profile, local_model_cost_map):
    """GPT-5.6 is served by Converse on bedrock-runtime, never by Invoke."""
    assert BedrockModelInfo.get_bedrock_route(f"bedrock/{profile.model_id}") == "converse"


@pytest.mark.parametrize("profile", GPT_5_6_PROFILES, ids=lambda p: p.model_id)
def test_bedrock_gpt_5_6_offers_tools_and_reasoning_effort_but_not_thinking(profile, local_model_cost_map):
    """GPT-5.x on Converse maps reasoning_effort to reasoning.effort, so reasoning_effort
    is offered while the Anthropic-only thinking/output_config are not, alongside the tool
    params these models accept."""
    supported = AmazonConverseConfig().get_supported_openai_params(model=f"bedrock/{profile.model_id}")

    assert "tools" in supported
    assert "tool_choice" in supported
    assert "reasoning_effort" in supported
    assert "thinking" not in supported
    assert "output_config" not in supported


# Cache-read prices are the `*-cache-read-input-tokens` usagetype rows of the AWS Price List API, us-east-1,
# https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json on 2026-09-15
