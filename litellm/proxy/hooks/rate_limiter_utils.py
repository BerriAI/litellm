"""
Shared utility functions for rate limiter hooks.
"""

from typing import Optional, Tuple, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import PriorityReservationDict

PROXY_LLM_PROVIDER_FALLBACK = "litellm_proxy"


def resolve_llm_provider_for_rate_limit(
    model: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve ``(model, llm_provider)`` for a request being rejected by an
    internal proxy-side rate-limit hook.

    These hooks fire from ``async_pre_call_hook`` — well before
    :func:`litellm.get_llm_provider` is invoked anywhere else in the request
    lifecycle — so the raised 429 would otherwise have an empty
    ``llm_provider`` field, making the resulting Prometheus
    ``litellm_proxy_failed_requests_metric`` show up with
    ``exception_class="RateLimitError"`` and no provider attribution.

    Resolution order:

    1. ``litellm.get_llm_provider(model)`` — covers raw provider/model
       strings the SDK already understands (``"gpt-4o-mini"``,
       ``"anthropic/claude-3-5-sonnet"``, ``"bedrock/..."`` etc.).
    2. **Router alias fallback** — nearly every real proxy deployment
       routes through a router ``model_name`` alias (e.g.
       ``"tpm-locked"`` → ``litellm_params.model: openai/gpt-4o-mini``).
       ``get_llm_provider`` doesn't know router aliases, so without this
       step every alias call ended up labeled ``"litellm_proxy"``,
       defeating the field's purpose for the most common case.
    3. Defensive fallback to ``("", "litellm_proxy")`` — used only when
       ``model`` is missing, malformed, or both lookups fail. We never let
       a secondary exception escape and mask the rate-limit error we're
       trying to surface.
    """
    if not model:
        return "", PROXY_LLM_PROVIDER_FALLBACK
    try:
        resolved_model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model,
        )
        return (
            resolved_model or model,
            custom_llm_provider or PROXY_LLM_PROVIDER_FALLBACK,
        )
    except Exception as e:
        alias_resolution = _resolve_provider_from_router_alias(model)
        if alias_resolution is not None:
            return alias_resolution
        verbose_proxy_logger.debug(
            "rate_limiter_utils.resolve_llm_provider_for_rate_limit: "
            "could not resolve provider for model=%s, falling back to %s. err=%s",
            model,
            PROXY_LLM_PROVIDER_FALLBACK,
            str(e),
        )
        return model, PROXY_LLM_PROVIDER_FALLBACK


def _resolve_provider_from_router_alias(
    model: str,
) -> Optional[Tuple[str, str]]:
    """
    Resolve a router ``model_name`` alias to ``(underlying_model, provider)``
    by scanning the active router's ``model_list``.

    Returns ``None`` if the router isn't initialized, the alias isn't
    registered, the deployment has no usable ``litellm_params.model``, or
    any underlying lookup raises. Callers fall through to the defensive
    ``litellm_proxy`` fallback in that case — never raising secondary
    exceptions out of the rate-limit raise path.
    """
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:
        return None
    if llm_router is None:
        return None
    try:
        model_list = getattr(llm_router, "model_list", None)
        if not model_list:
            return None
        for deployment in model_list:
            if not isinstance(deployment, dict):
                continue
            if deployment.get("model_name") != model:
                continue
            params = deployment.get("litellm_params")
            if not isinstance(params, dict):
                continue
            underlying_model = params.get("model")
            if not isinstance(underlying_model, str) or not underlying_model:
                continue
            try:
                resolved_model, custom_llm_provider, _, _ = litellm.get_llm_provider(
                    model=underlying_model,
                )
            except Exception:
                continue
            if not custom_llm_provider:
                continue
            # Prefer the underlying provider-qualified model so the failure
            # callback / Prometheus label points at the actual deployment, not
            # the alias.
            return (
                resolved_model or underlying_model,
                custom_llm_provider,
            )
        return None
    except Exception:
        return None


def convert_priority_to_percent(
    value: Union[float, PriorityReservationDict], model_info: Optional[ModelGroupInfo]
) -> float:
    """
    Convert priority reservation value to percentage (0.0-1.0).

    Supports three formats:
    1. Plain float/int: 0.9 -> 0.9 (90%)
    2. Dict with percent: {"type": "percent", "value": 0.9} -> 0.9
    3. Dict with rpm: {"type": "rpm", "value": 900} -> 900/model_rpm
    4. Dict with tpm: {"type": "tpm", "value": 900000} -> 900000/model_tpm

    Args:
        value: Priority value as float or dict with type/value keys
        model_info: Model configuration containing rpm/tpm limits

    Returns:
        float: Percentage value between 0.0 and 1.0
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        val_type = value.get("type", "percent")
        val_num = value.get("value", 1.0)

        if val_type == "percent":
            return float(val_num)
        elif val_type == "rpm" and model_info and model_info.rpm and model_info.rpm > 0:
            return float(val_num) / model_info.rpm
        elif val_type == "tpm" and model_info and model_info.tpm and model_info.tpm > 0:
            return float(val_num) / model_info.tpm

        # Fallback: treat as percent
        return float(val_num)
