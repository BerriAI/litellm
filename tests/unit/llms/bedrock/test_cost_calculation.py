import json
from importlib.resources import files
from typing import Final

import pytest

import litellm
from litellm.types.utils import Usage

CLAUDE_PLATFORM_PREFIX: Final = "bedrock/claude_platform/"
CLAUDE_PLATFORM_KEYS: Final = tuple(
    key
    for key in json.loads(files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text())
    if key.startswith(CLAUDE_PLATFORM_PREFIX)
)


def test_claude_platform_rows_exist() -> None:
    assert CLAUDE_PLATFORM_KEYS


@pytest.mark.parametrize("inference_geo", ["global", "us"])
@pytest.mark.parametrize("key", CLAUDE_PLATFORM_KEYS)
def test_claude_platform_cost_matches_claude_api_cost(
    local_model_cost_map: None, key: str, inference_geo: str
) -> None:
    usage: Final = Usage(
        prompt_tokens=1_000,
        completion_tokens=200,
        cache_read_input_tokens=300,
        cache_creation_input_tokens=250,
        inference_geo=inference_geo,
    )

    claude_platform_cost: Final = litellm.cost_per_token(
        model=key.removeprefix("bedrock/"), custom_llm_provider="bedrock", usage_object=usage
    )
    claude_api_cost: Final = litellm.cost_per_token(
        model=key.removeprefix(CLAUDE_PLATFORM_PREFIX), custom_llm_provider="anthropic", usage_object=usage
    )

    assert claude_platform_cost == pytest.approx(claude_api_cost)
    assert sum(claude_platform_cost) > 0


def test_an_unreadable_geo_multiplier_raises_as_on_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing catches the geo ``float()`` on Bedrock, so ``completion_cost`` moves on to its next candidate."""
    from litellm.llms.bedrock.cost_calculation import cost_per_token

    row: Final = {
        "litellm_provider": "bedrock",
        "mode": "chat",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 2.5e-05,
        "provider_specific_entry": {"us": "high"},
    }
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost, "anthropic.claude-probe-geo": row})

    with pytest.raises(ValueError, match=r"No price for model=anthropic\.claude-probe-geo"):
        cost_per_token(
            model="anthropic.claude-probe-geo",
            usage=Usage(prompt_tokens=1000, completion_tokens=1000, inference_geo="us"),
        )
