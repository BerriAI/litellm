"""
Helper util for handling openai-specific cost calculation
- e.g.: prompt caching
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    TranscriptionDocument,
    VideoDocument,
    branch_cost,
    price_units,
    prompt_and_completion,
)
from litellm.rust_bridge._native import NativeReceipt
from litellm.types.utils import CallTypes, ModelInfo, Usage
from litellm.utils import get_model_info


def cost_router(call_type: CallTypes) -> Literal["cost_per_token", "cost_per_second"]:
    if call_type == CallTypes.atranscription or call_type == CallTypes.transcription:
        return "cost_per_second"
    else:
        return "cost_per_token"


def cost_per_token(
    model: str,
    usage: Usage,
    service_tier: str | None = None,
    data_residency: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information
        - data_residency: optional OpenAI data-residency mode (e.g. "eu", "us", "fedramp"),
          inferred from api_base, including gov.api.openai.com. Applies the
          model's regional-processing uplift multiplier when set.

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return branch_cost(
        model,
        "openai",
        usage,
        get_model_info(model=model, custom_llm_provider="openai"),
        service_tier=service_tier,
        data_residency=data_residency,
    )


def cost_per_second(model: str, custom_llm_provider: str | None, duration: float = 0.0) -> tuple[float, float]:
    """
    Calculates the cost per second for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, the custom llm provider
        - duration: float, the duration of the response in seconds

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """

    provider: Final = custom_llm_provider or "openai"
    return prompt_and_completion(
        price_units(
            TranscriptionDocument(kind="transcription", model=model, provider=provider, seconds=duration),
            catalog=get_model_info(model=model, custom_llm_provider=provider),
        )
    )


def clip_seconds(duration: object) -> float | None:
    """A request's ``duration`` as the positive, finite seconds a clip table cell is rounded from, else ``None``."""
    if isinstance(duration, bool) or not isinstance(duration, (str, int, float)):
        return None
    try:
        seconds: Final = float(duration)
    except (ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def video_clip_cost(model_info: Mapping[str, object], duration: object, resolution: object) -> float | None:
    """The ``output_cost_per_video_clip`` cell for ``duration`` rounded to whole seconds and ``resolution``, ``None``
    when the table has no such cell."""
    seconds: Final = clip_seconds(duration)
    if seconds is None or not isinstance(resolution, str):
        return None
    outcome: Final = price_units(
        VideoDocument(
            kind="video",
            model="video_clip",
            provider=None,
            seconds=seconds,
            resolution=resolution,
            videos=1,
            input_images=0,
            deployment=False,
            requested_row=False,
        ),
        catalog=MappingProxyType({"output_cost_per_video_clip": model_info.get("output_cost_per_video_clip")}),
    )
    return outcome.total if isinstance(outcome, NativeReceipt) else None


def priced_video(
    model: str,
    custom_llm_provider: str | None,
    seconds: float | None,
    video_resolution: str | None,
    deployment: Mapping[str, object] | None,
    catalog: Mapping[str, object] | None,
    videos: int = 1,
    input_images: int = 0,
    requested_row: Mapping[str, object] | None = None,
) -> float:
    """The cost of ``videos`` videos and the billable input images. ``deployment`` replaces ``catalog``, the row the
    caller looked up for the model, and ``requested_row`` is read for an input image rate the deployment lacks."""
    outcome: Final = price_units(
        VideoDocument(
            kind="video",
            model=model,
            provider=custom_llm_provider,
            seconds=seconds,
            resolution=video_resolution.strip().lower() if video_resolution is not None else None,
            videos=videos,
            input_images=input_images,
            deployment=deployment is not None,
            requested_row=requested_row is not None,
        ),
        tuple(table for table in (deployment, requested_row) if table is not None),
        catalog,
    )
    if isinstance(outcome, NativeReceipt):
        return outcome.total
    verbose_logger.warning(
        "No cost information found for video model %s. Please add pricing to model_prices_and_context_window.json",
        model,
    )
    return 0.0


def video_generation_cost(
    model: str,
    duration_seconds: float,
    custom_llm_provider: str | None = None,
    model_info: ModelInfo | None = None,
    video_resolution: str | None = None,
) -> float:
    """
    Calculates the cost for video generation based on duration in seconds.

    Input:
        - model: str, the model name without provider prefix
        - duration_seconds: float, the duration of the generated video in seconds
        - custom_llm_provider: str, the custom llm provider
        - model_info: Optional[dict], deployment-level model info containing
            custom video pricing. When provided, skips the global
            get_model_info() lookup so that deployment-specific pricing is used.
        - video_resolution: Optional resolution label from usage (e.g. ``720p``, ``1080p``).

    Returns:
        float - total_cost_in_usd
    """
    return priced_video(
        model=model,
        custom_llm_provider=custom_llm_provider,
        seconds=duration_seconds,
        video_resolution=video_resolution,
        deployment=model_info,
        catalog=None
        if model_info is not None
        else get_model_info(model=model, custom_llm_provider=custom_llm_provider or "openai"),
    )
