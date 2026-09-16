import os
from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest


def _skip_live_prompt_caching_test():
    if os.environ.get("LITELLM_RUN_LIVE_PROMPT_CACHING_TESTS") != "1":
        pytest.skip("Live prompt-caching E2E tests are opt-in")
    if os.environ.get("CASSETTE_REDIS_URL"):
        pytest.skip("Live prompt-caching E2E tests cannot run under VCR replay")


def cheapest_together_chat_model(*capability_flags: str) -> str:
    import litellm

    today = date.today().isoformat()

    def qualifies(name: str, entry: Mapping[str, Any]) -> bool:
        deprecation_date = entry.get("deprecation_date")
        return (
            name.startswith("together_ai/")
            and entry.get("litellm_provider") == "together_ai"
            and entry.get("mode") == "chat"
            and (deprecation_date is None or deprecation_date > today)
            and (entry.get("input_cost_per_token") or 0.0) > 0
            and (entry.get("output_cost_per_token") or 0.0) > 0
            and all(bool(entry.get(flag)) for flag in capability_flags)
        )

    candidates = sorted(
        (
            name
            for name, entry in litellm.model_cost.items()
            if isinstance(entry, Mapping) and qualifies(name, entry)
        ),
        key=lambda name: (
            litellm.model_cost[name].get("input_cost_per_token") or 0.0,
            litellm.model_cost[name].get("output_cost_per_token") or 0.0,
            name,
        ),
    )
    assert candidates, f"no live together_ai chat model in the cost map satisfies {capability_flags}"
    return candidates[0]
