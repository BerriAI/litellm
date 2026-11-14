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
