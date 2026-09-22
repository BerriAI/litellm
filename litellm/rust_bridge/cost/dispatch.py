from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Final, ParamSpec, TypeVar

from typing_extensions import Never, assert_never

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import CostContext, Rules, decision
from litellm.rust_bridge.configuration import Decision

P: Final = ParamSpec("P")
R: Final = TypeVar("R")


class CostApi(str, Enum):
    COST_PER_TOKEN = "cost_per_token"
    COMPLETION_COST = "completion_cost"
    RESPONSE_COST_CALCULATOR = "response_cost_calculator"
    GENERIC_COST_PER_TOKEN = "generic_cost_per_token"
    REPLICATE_COMPLETION_PRICING = "get_replicate_completion_pricing"
    OCR_COST = "ocr_cost"
    OCR_BATCH_COST = "ocr_batch_cost"
    VECTOR_STORE_SEARCH_COST = "vector_store_search_cost"
    RERANK_COST = "rerank_cost"
    TRANSCRIPTION_COST = "transcription_cost"
    IMAGE_COST = "default_image_cost_calculator"
    VIDEO_COST = "default_video_cost_calculator"
    BATCH_COST = "batch_cost_calculator"
    REALTIME_STREAM_COST = "handle_realtime_stream_cost_calculation"
    REALTIME_TRANSCRIPTION_COST = "handle_realtime_transcription_cost_calculation"
    SELECT_COST_METRIC = "select_cost_metric_for_model"
    BATCH_COST_RATES = "get_batch_cost_rates"
    BILLED_TOKEN_RATES = "get_billed_token_rates"
    TOKEN_TYPE_COST_BREAKDOWN = "get_token_type_cost_breakdown"
    PROMPT_CACHING_SAVINGS = "calculate_prompt_caching_savings"
    CACHE_WRITING_COST = "calculate_cache_writing_cost"
    COST_COMPONENT = "calculate_cost_component"
    IMAGE_RESPONSE_COST_FROM_USAGE = "calculate_image_response_cost_from_usage"
    IMAGE_RESPONSE_WEB_SEARCH_COST = "calculate_image_response_web_search_cost"


def _native_cost_api(value: object) -> Callable[[str], Never] | None:
    if not callable(value):
        return None

    def invoke(name: str) -> Never:
        value(name)
        raise RuntimeError(f"Rust cost API {name} returned without a result")

    return invoke


NATIVE_COST_API: Final = NativeBinding("cost_api", validate=_native_cost_api)
COST_CONTEXT: Final = CostContext()


def run_cost_api(
    api: CostApi,
    python: Callable[[], R],
    *,
    rules: Rules | None = None,
    binding: NativeBinding[Callable[[str], Never]] = NATIVE_COST_API,
) -> R:
    selected: Final = decision(COST_CONTEXT, rules)
    match selected:
        case Decision.PYTHON:
            return python()
        case Decision.RUST_REQUIRED:
            native: Final = binding.load()
            if native is None:
                raise RuntimeError(f"Rust cost API {api.value} is unavailable")
            return native(api.value)
        case Decision.RUST_WITH_FALLBACK:
            raise RuntimeError("Cost policy must be PYTHON_ONLY or RUST_REQUIRED")
        case _:
            assert_never(selected)


def route_cost_api(api: CostApi) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(python: Callable[P, R]) -> Callable[P, R]:
        @wraps(python)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:  # kwargs-ok: ParamSpec preserves the wrapped signature
            return run_cost_api(api, lambda: python(*args, **kwargs))

        return wrapped

    return decorate
