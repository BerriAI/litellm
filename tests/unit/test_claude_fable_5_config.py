"""
Validate Claude Fable 5, Claude Fable 5.1 and Claude Fable 5.2 model configuration entries.

Fable 5 is a new tier above Opus ($10/$50 per MTok) with the same adaptive-only
API surface as Opus 4.7/4.8. The cost-map entries below are what make the model
resolvable across Anthropic, Bedrock, Vertex AI, and Azure AI (Microsoft
Foundry), and the ``supports_adaptive_thinking`` flag is what makes LiteLLM send
``thinking.type='adaptive'`` instead of the legacy ``enabled``/``budget_tokens``
shape, which Fable 5 rejects with a 400.
"""

import json
import os
from typing import Final

import pytest

import litellm
from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


def test_fable_5_registered_for_bedrock_converse():
    assert "anthropic.claude-fable-5" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5",
        "anthropic/claude-fable-5",
        "anthropic.claude-fable-5",
        "bedrock/us.anthropic.claude-fable-5",
        "bedrock/invoke/eu.anthropic.claude-fable-5",
        "bedrock/global.anthropic.claude-fable-5",
        "vertex_ai/claude-fable-5",
        "azure_ai/claude-fable-5",
    ],
)
def test_adaptive_thinking_detected_for_fable_5(local_model_cost_map, model):
    """Provider-routed ids must resolve to a flagged entry so ``reasoning_effort``
    maps to ``thinking.type='adaptive'`` + ``output_config.effort``."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo.is_adaptive_thinking_model(model, "anthropic") is True


FABLE_5_1_VARIANTS = (
    "claude-fable-5-1",
    "anthropic.claude-fable-5-1",
    "global.anthropic.claude-fable-5-1",
    "us.anthropic.claude-fable-5-1",
    "eu.anthropic.claude-fable-5-1",
    "vertex_ai/claude-fable-5-1",
    "vertex_ai/claude-fable-5-1@default",
    "azure_ai/claude-fable-5-1",
)


def test_fable_5_1_present_in_bundled_backup():
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    for model_name in FABLE_5_1_VARIANTS:
        assert model_name in backup, f"Missing from backup cost map: {model_name}"
        assert backup[model_name] == root[model_name], model_name


def test_fable_5_1_registered_for_bedrock_converse():
    assert "anthropic.claude-fable-5-1" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-1",
        "anthropic/claude-fable-5-1",
        "anthropic.claude-fable-5-1",
        "bedrock/us.anthropic.claude-fable-5-1",
        "bedrock/invoke/eu.anthropic.claude-fable-5-1",
        "bedrock/global.anthropic.claude-fable-5-1",
        "vertex_ai/claude-fable-5-1",
        "azure_ai/claude-fable-5-1",
    ],
)
def test_adaptive_thinking_detected_for_fable_5_1(local_model_cost_map, model):
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo.is_adaptive_thinking_model(model, "anthropic") is True


FABLE_5_2_VARIANTS = (
    "claude-fable-5-2",
    "anthropic.claude-fable-5-2",
    "global.anthropic.claude-fable-5-2",
    "us.anthropic.claude-fable-5-2",
    "eu.anthropic.claude-fable-5-2",
    "us-gov.anthropic.claude-fable-5-2",
    "bedrock/us-gov-west-1/anthropic.claude-fable-5-2",
    "bedrock/us-gov-east-1/anthropic.claude-fable-5-2",
    "vertex_ai/claude-fable-5-2",
    "vertex_ai/claude-fable-5-2@default",
    "bedrock/claude_platform/claude-fable-5-2",
)


def test_fable_5_2_present_in_bundled_backup() -> None:
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    for model_name in FABLE_5_2_VARIANTS:
        assert model_name in backup, f"Missing from backup cost map: {model_name}"
        assert backup[model_name] == root[model_name], model_name


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5-2",
        "anthropic/claude-fable-5-2",
        "claude-fable-latest",
        "anthropic.claude-fable-5-2",
        "bedrock/us.anthropic.claude-fable-5-2",
        "bedrock/invoke/eu.anthropic.claude-fable-5-2",
        "bedrock/global.anthropic.claude-fable-5-2",
        "bedrock/claude-fable-latest",
        "vertex_ai/claude-fable-5-2",
        "vertex_ai/claude-fable-5-2@default",
        "bedrock/claude_platform/claude-fable-5-2",
    ],
)
def test_adaptive_thinking_detected_for_fable_5_2(local_model_cost_map, model: str) -> None:
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo.is_adaptive_thinking_model(model, "anthropic") is True


def test_fable_5_2_registered_for_bedrock_converse() -> None:
    assert "anthropic.claude-fable-5-2" in BEDROCK_CONVERSE_MODELS


def test_fable_5_2_fast_mode_doubles_every_token_price_on_the_anthropic_api(local_model_cost_map) -> None:
    """Anthropic bills fast mode at 2x the standard rate on every token type
    (platform.claude.com/docs/en/about-claude/pricing, read 2026-11-09), and the
    Anthropic cost path is the one that reads the row's `fast` multiplier."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_provider_specific_speed_multiplier
    from litellm.types.utils import Usage

    info: Final = litellm.get_model_info("claude-fable-5-2")
    assert get_provider_specific_speed_multiplier(info, Usage(speed="fast")) == 2.0
    assert get_provider_specific_speed_multiplier(info, Usage()) == 1.0


def test_fable_5_2_vertex_rows_do_not_advertise_fast_mode(local_model_cost_map) -> None:
    """Vertex cost calculation ignores the `fast` multiplier, so a Vertex row that
    advertised fast mode would bill fast calls at the standard rate."""
    for model in ("vertex_ai/claude-fable-5-2", "vertex_ai/claude-fable-5-2@default"):
        info: Final = litellm.get_model_info(model)
        assert info.get("supports_fast_mode") is not True, model
        assert "fast" not in (info.get("provider_specific_entry") or {}), model


def test_fable_5_2_accepts_the_speed_param_on_the_anthropic_api(local_model_cost_map) -> None:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo.supports_fast_mode("claude-fable-5-2", "anthropic") is True
    assert AnthropicConfig._model_supports_speed_param("claude-fable-5-2", "anthropic") is True


MYTHOS_5_2_VERTEX_ROWS: Final = (
    ("vertex_ai/claude-mythos-5-2", "vertex_ai/claude-fable-5-2"),
    ("vertex_ai/claude-mythos-5-2@default", "vertex_ai/claude-fable-5-2@default"),
)


def _without_source(row: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in row.items() if k != "source"}


def test_mythos_5_2_vertex_rows_present_in_bundled_backup() -> None:
    backup: Final = GetModelCostMap.load_local_model_cost_map()
    root: Final = _load_root_cost_map()
    for model_name, _ in MYTHOS_5_2_VERTEX_ROWS:
        assert model_name in backup, f"Missing from backup cost map: {model_name}"
        assert backup[model_name] == root[model_name], model_name


@pytest.mark.parametrize(("mythos_row", "fable_row"), MYTHOS_5_2_VERTEX_ROWS)
def test_mythos_5_2_vertex_rows_match_fable_5_2(mythos_row: str, fable_row: str) -> None:
    """Anthropic ships Mythos 5.2 as a verified-access twin of Fable 5.2 with "the same
    specifications and pricing" (platform.claude.com/docs/en/models/mythos-5-2/overview,
    read 2026-11-10), so the Vertex rows must stay identical apart from their source."""
    root: Final = _load_root_cost_map()
    assert _without_source(root[mythos_row]) == _without_source(root[fable_row])
    assert root[mythos_row]["source"] == "https://platform.claude.com/docs/en/models/mythos-5-2/overview"


@pytest.mark.parametrize("model", [row for row, _ in MYTHOS_5_2_VERTEX_ROWS])
def test_adaptive_thinking_detected_for_mythos_5_2_on_vertex(local_model_cost_map, model: str) -> None:
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo.is_adaptive_thinking_model(model, "anthropic") is True


def _vertex_claude_cached_usage() -> Usage:
    return Usage(
        prompt_tokens=1600,
        completion_tokens=50,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=300, cache_creation_tokens=300),
        cache_read_input_tokens=300,
        cache_creation_input_tokens=300,
        cache_creation_token_details=CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=200, ephemeral_1h_input_tokens=100
        ),
    )


@pytest.mark.parametrize(("mythos_row", "fable_row"), MYTHOS_5_2_VERTEX_ROWS)
def test_mythos_5_2_on_vertex_bills_like_fable_5_2(local_model_cost_map, mythos_row: str, fable_row: str) -> None:
    """Billing through the Vertex cost path, with cache reads, 5m and 1h cache writes and the
    regional-endpoint uplift, must charge Mythos 5.2 exactly what Fable 5.2 is charged."""
    for location in ("global", "us-east5"):
        mythos_cost: Final = litellm.cost_per_token(
            model=mythos_row,
            custom_llm_provider="vertex_ai",
            usage_object=_vertex_claude_cached_usage(),
            vertex_location=location,
        )
        fable_cost: Final = litellm.cost_per_token(
            model=fable_row,
            custom_llm_provider="vertex_ai",
            usage_object=_vertex_claude_cached_usage(),
            vertex_location=location,
        )
        assert mythos_cost == fable_cost, location
        assert mythos_cost[0] > 0 and mythos_cost[1] > 0, location


@pytest.mark.parametrize("model", [row for row, _ in MYTHOS_5_2_VERTEX_ROWS])
def test_mythos_5_2_on_vertex_applies_regional_uplift(local_model_cost_map, model: str) -> None:
    uplift: Final = _load_root_cost_map()[model]["regional_endpoint_uplift_multiplier"]
    global_cost: Final = litellm.cost_per_token(
        model=model,
        custom_llm_provider="vertex_ai",
        usage_object=_vertex_claude_cached_usage(),
        vertex_location="global",
    )
    regional_cost: Final = litellm.cost_per_token(
        model=model,
        custom_llm_provider="vertex_ai",
        usage_object=_vertex_claude_cached_usage(),
        vertex_location="us-east5",
    )
    assert regional_cost[0] == pytest.approx(global_cost[0] * uplift)
    assert regional_cost[1] == pytest.approx(global_cost[1] * uplift)
