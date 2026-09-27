import os
from datetime import date

import pytest
from pydantic import BaseModel, ConfigDict


def _skip_live_prompt_caching_test():
    if os.environ.get("LITELLM_RUN_LIVE_PROMPT_CACHING_TESTS") != "1":
        pytest.skip("Live prompt-caching E2E tests are opt-in")
    if os.environ.get("CASSETTE_REDIS_URL"):
        pytest.skip("Live prompt-caching E2E tests cannot run under VCR replay")



class TogetherCostEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_provider: str | None = None
    mode: str | None = None
    deprecation_date: str | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    supports_function_calling: bool | None = None
    supports_response_schema: bool | None = None


def cheapest_together_chat_model(
    *, function_calling: bool = False, response_schema: bool = False
) -> str:
    import litellm

    today = date.today().isoformat()

    def qualifies(name: str, entry: TogetherCostEntry) -> bool:
        return (
            name.startswith("together_ai/")
            and entry.litellm_provider == "together_ai"
            and entry.mode == "chat"
            and (entry.deprecation_date is None or entry.deprecation_date > today)
            and (entry.input_cost_per_token or 0.0) > 0
            and (entry.output_cost_per_token or 0.0) > 0
            and (not function_calling or bool(entry.supports_function_calling))
            and (not response_schema or bool(entry.supports_response_schema))
        )

    registry: dict[str, TogetherCostEntry] = {
        name: TogetherCostEntry.model_validate(raw)
        for name, raw in litellm.model_cost.items()
        if isinstance(raw, dict) and name.startswith("together_ai/")
    }
    candidates = sorted(
        (name for name, entry in registry.items() if qualifies(name, entry)),
        key=lambda name: (
            registry[name].input_cost_per_token or 0.0,
            registry[name].output_cost_per_token or 0.0,
            name,
        ),
    )
    assert candidates, (
        "no live together_ai chat model in the cost map satisfies "
        f"function_calling={function_calling} response_schema={response_schema}"
    )
    return candidates[0]
