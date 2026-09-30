from collections.abc import Mapping
from datetime import datetime, timezone
from math import isfinite
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.cost_calculator import (
    _select_model_name_for_cost_calc,  # pyright: ignore[reportPrivateUsage]  # shares completion_cost's deployment tariff selection
    completion_cost,  # pyright: ignore[reportUnknownVariableType]  # legacy optional parameters are untyped
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.management_endpoints.prompt_cache_prediction import CacheTokenBuckets
from litellm.types.utils import CacheCreationTokenDetails, ModelResponse, PromptTokensDetailsWrapper, Usage

_PRICE_ENTRY: Final = TypeAdapter(Mapping[str, object])


def _valid_price(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and value >= 0


def _has_required_prices(prices: Mapping[str, object], tokens: CacheTokenBuckets, completion_tokens: int = 0) -> bool:
    required: Final = (
        ("input_cost_per_token", True),
        ("cache_read_input_token_cost", tokens.cache_read_input_tokens > 0),
        ("cache_creation_input_token_cost", tokens.cache_creation_5m_input_tokens > 0),
        ("cache_creation_input_token_cost_above_1hr", tokens.cache_creation_1h_input_tokens > 0),
        ("output_cost_per_token", completion_tokens > 0),
    )
    if any(needed and not _valid_price(prices.get(key)) for key, needed in required):
        return False
    return all(
        _valid_price(value)
        for key, value in prices.items()
        if value is not None and any(needed and key.startswith(f"{base}_above_") for base, needed in required)
    )


def price_cache_tokens(
    model: str, deployment_id: str, tokens: CacheTokenBuckets, completion_tokens: int = 0
) -> float | None:
    try:
        selected_model: Final = _select_model_name_for_cost_calc(
            model=model,
            completion_response=None,
            custom_pricing=True,
            custom_llm_provider="anthropic",
            router_model_id=deployment_id,
        )
        if selected_model is None:
            return None
        model_info: Final = litellm.get_model_info(model=selected_model, custom_llm_provider="anthropic")
        registry: Final = _PRICE_ENTRY.validate_python(litellm.model_cost)  # pyright: ignore[reportUnknownMemberType]  # legacy registry is validated at this boundary
        price_entry: Final = registry.get(model_info["key"])
        if price_entry is None:
            return None
        prices: Final = _PRICE_ENTRY.validate_python(price_entry)
        if completion_tokens < 0 or not _has_required_prices(prices, tokens, completion_tokens):
            return None
        usage: Final = Usage(
            prompt_tokens=tokens.total_tokens,
            completion_tokens=completion_tokens,
            total_tokens=tokens.total_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=tokens.cache_read_input_tokens,
                cache_creation_tokens=tokens.cache_creation_5m_input_tokens + tokens.cache_creation_1h_input_tokens,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=tokens.cache_creation_5m_input_tokens,
                    ephemeral_1h_input_tokens=tokens.cache_creation_1h_input_tokens,
                ),
            ),
        )
        logging_obj: Final = Logging(
            model=model,
            messages=[],  # mutable-ok: Logging requires a list
            stream=False,
            call_type="completion",
            start_time=datetime.now(timezone.utc),
            litellm_call_id="prompt-cache-prediction",
            function_id="prompt-cache-prediction",
        )
        completion_cost(
            completion_response=ModelResponse(model=model, usage=usage),
            model=model,
            custom_llm_provider="anthropic",
            custom_pricing=True,
            router_model_id=deployment_id,
            litellm_logging_obj=logging_obj,
        )
        breakdown: Final = logging_obj.cost_breakdown
        input_cost: Final = breakdown.get("input_cost") if breakdown is not None else None
        output_cost: Final = breakdown.get("output_cost") if breakdown is not None else None
        if input_cost is None or output_cost is None:
            return None
        cost: Final = input_cost + output_cost
        return cost if _valid_price(cost) else None
    except Exception:  # noqa: BLE001  # the shared pricing owners raise plain Exception for unpriceable models
        return None
