from typing import Final

import pytest

import litellm
from litellm.utils import supports_prompt_caching

MODEL: Final = "vertex_ai/xai/grok-4.6"
GROK_KEY_PREFIXES: Final = ("vertex_ai/xai/grok-", "azure_ai/grok-", "xai/grok-")


@pytest.mark.usefixtures("local_model_cost_map")
def test_grok_models_with_cache_read_price_advertise_prompt_caching() -> None:
    cached_grok_models = tuple(
        key
        for key, entry in litellm.model_cost.items()
        if key.startswith(GROK_KEY_PREFIXES) and entry.get("cache_read_input_token_cost")
    )
    assert cached_grok_models, "expected at least one grok model with a cache read price"

    missing_flag = tuple(key for key in cached_grok_models if supports_prompt_caching(model=key) is not True)
    assert missing_flag == (), (
        f"grok models with cache_read_input_token_cost fail supports_prompt_caching: {missing_flag}"
    )


