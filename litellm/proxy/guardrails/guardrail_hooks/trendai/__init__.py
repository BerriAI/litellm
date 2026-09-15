from typing import TYPE_CHECKING, Final

import litellm
from litellm.types.guardrails import GuardrailEventHooks, Mode

from ._models import TrendAISettings
from .trendai import GUARDRAIL_NAME, TrendAIGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def _normalize_event_hook(
    mode: str | list[str] | Mode,
) -> GuardrailEventHooks | list[GuardrailEventHooks] | Mode:
    if isinstance(mode, str):
        return GuardrailEventHooks(mode)
    if isinstance(mode, list):
        return [GuardrailEventHooks(item) for item in mode]
    return mode


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> TrendAIGuardrail:
    settings: Final = TrendAISettings.model_validate(litellm_params.model_dump(mode="python"))
    guardrail_name: Final = guardrail["guardrail_name"]
    callback: Final = TrendAIGuardrail(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        app_name=settings.app_name,
        fallback_on_error=settings.fallback_on_error,
        timeout=settings.timeout,
        stream_batch_size=settings.stream_batch_size,
        stream_overlap_size=settings.stream_overlap_size,
        response_content_chunk_size_bytes=settings.response_content_chunk_size_bytes,
        logging_only_scan=settings.logging_only_scan,
        guardrail_name=guardrail_name,
        event_hook=_normalize_event_hook(litellm_params.mode),
        default_on=litellm_params.default_on is True,
    )
    litellm.logging_callback_manager.add_litellm_callback(  # pyright: ignore[reportUnknownMemberType]  # callback union is partially untyped
        callback
    )
    return callback


guardrail_initializer_registry: Final = {GUARDRAIL_NAME: initialize_guardrail}
guardrail_class_registry: Final = {GUARDRAIL_NAME: TrendAIGuardrail}

__all__ = ("TrendAIGuardrail",)
