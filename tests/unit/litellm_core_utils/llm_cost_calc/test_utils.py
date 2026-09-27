import asyncio
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

TIER_MODEL: Final = "tier-priced-test-model"
TIER_ROW: Final[Mapping[str, float]] = MappingProxyType(
    {
        "input_cost_per_token": 4e-06,
        "output_cost_per_token": 8e-06,
        "cache_read_input_token_cost": 1e-06,
        "input_cost_per_token_flex": 1e-06,
        "output_cost_per_token_flex": 2e-06,
        "cache_read_input_token_cost_flex": 2.5e-07,
        "input_cost_per_token_balanced": 2e-06,
        "output_cost_per_token_balanced": 4e-06,
        "cache_read_input_token_cost_balanced": 5e-07,
    }
)
PROMPT_TOKENS: Final = 1000
CACHED_TOKENS: Final = 200
COMPLETION_TOKENS: Final = 500
TIER_API_BASE: Final = "https://tier-pricing.invalid/v1"


def _cost_at(prices: Mapping[str, float], column_suffix: str) -> float:
    return (
        (PROMPT_TOKENS - CACHED_TOKENS) * prices[f"input_cost_per_token{column_suffix}"]
        + CACHED_TOKENS * prices[f"cache_read_input_token_cost{column_suffix}"]
        + COMPLETION_TOKENS * prices[f"output_cost_per_token{column_suffix}"]
    )


def _register_tier_model() -> None:
    litellm.register_model({TIER_MODEL: {"litellm_provider": "openai", "mode": "chat", **TIER_ROW}})


@pytest.mark.parametrize(
    ("service_tier", "column_suffix"),
    [
        pytest.param(None, "", id="no-tier-bills-base"),
        pytest.param("auto", "", id="auto-bills-base"),
        pytest.param("default", "", id="default-bills-base"),
        pytest.param("priority", "", id="tier-without-columns-bills-base"),
        pytest.param("flex", "_flex", id="flex"),
        pytest.param("balanced", "_balanced", id="balanced"),
        pytest.param("BALANCED", "_balanced", id="balanced-any-case"),
    ],
)
def test_completion_cost_bills_the_price_columns_of_the_service_tier(
    local_model_cost_map: None, service_tier: str | None, column_suffix: str
) -> None:
    _register_tier_model()
    response: Final = ModelResponse(
        model=TIER_MODEL,
        usage=Usage(
            prompt_tokens=PROMPT_TOKENS,
            completion_tokens=COMPLETION_TOKENS,
            total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=CACHED_TOKENS),
        ),
    )

    cost: Final = litellm.completion_cost(
        completion_response=response, model=TIER_MODEL, custom_llm_provider="openai", service_tier=service_tier
    )

    assert cost == pytest.approx(_cost_at(TIER_ROW, column_suffix))


class _CostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.cost_by_model_group: Mapping[str, float] = MappingProxyType({})

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        payload: Final = kwargs.get("standard_logging_object")
        cost: Final = kwargs.get("response_cost")
        if isinstance(payload, dict) and isinstance(cost, float):
            self.cost_by_model_group = MappingProxyType(
                {**self.cost_by_model_group, str(payload.get("model_group")): cost}
            )


async def _logged_cost(recorder: _CostRecorder, model_group: str) -> float:
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    assert model_group in recorder.cost_by_model_group, recorder.cost_by_model_group
    return recorder.cost_by_model_group[model_group]


def _chat_completion_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-tier",
        "object": "chat.completion",
        "created": 0,
        "model": TIER_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            "prompt_tokens_details": {"cached_tokens": CACHED_TOKENS},
        },
    }


DEPLOYMENT_OVERRIDE: Final = 9e-06
PRICE_COLUMNS: Final = ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost")
PARITY_ROW: Final[Mapping[str, float]] = MappingProxyType(
    {
        "input_cost_per_token": 4e-06,
        "output_cost_per_token": 8e-06,
        "cache_read_input_token_cost": 1e-06,
        **{
            f"{column}_{tier}": price
            for tier in ("flex", "balanced")
            for column, price in zip(PRICE_COLUMNS, (1e-06, 2e-06, 2.5e-07), strict=True)
        },
    }
)


@pytest.mark.parametrize(
    "overridden_columns",
    [
        pytest.param((), id="catalog-only"),
        *(pytest.param((column,), id=f"deployment-overrides-{column}") for column in PRICE_COLUMNS),
        pytest.param(PRICE_COLUMNS, id="deployment-overrides-all"),
    ],
)
@pytest.mark.asyncio
async def test_router_prices_balanced_columns_by_the_same_rules_as_flex(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    overridden_columns: tuple[str, ...],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    recorder: Final = _CostRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    litellm.register_model({TIER_MODEL: {"litellm_provider": "openai", "mode": "chat", **PARITY_ROW}})
    respx_mock.post(f"{TIER_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_completion_body())
    )
    group: Final = {tier: f"{tier}-{uuid.uuid4().hex}" for tier in ("flex", "balanced")}
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": group[tier],
                "litellm_params": {
                    "model": f"openai/{TIER_MODEL}",
                    "api_key": "sk-test",
                    "api_base": TIER_API_BASE,
                    **{f"{column}_{tier}": DEPLOYMENT_OVERRIDE for column in overridden_columns},
                },
            }
            for tier in ("flex", "balanced")
        ]
    )

    for tier in ("flex", "balanced"):
        await router.acompletion(model=group[tier], messages=[{"role": "user", "content": "hi"}], service_tier=tier)
    flex_cost: Final = await _logged_cost(recorder, group["flex"])
    balanced_cost: Final = await _logged_cost(recorder, group["balanced"])

    assert balanced_cost == pytest.approx(flex_cost)
    if not overridden_columns:
        assert balanced_cost == pytest.approx(_cost_at(PARITY_ROW, "_balanced"))
