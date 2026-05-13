"""
Deployment affinity for prompt-caching-related routing.

1. **Anthropic-style breakpoints**: When messages include ``cache_control`` blocks
   (ephemeral), uses :class:`PromptCachingCache` (SHA prefix of the cacheable prefix)
   to pin ``model_id``. Redis keys look like ``deployment:<hash>:prompt_caching``.

2. **OpenAI prompt-cache affinity** (optional): For ``custom_llm_provider == "openai"``
   and a non-empty ``prompt_cache_key`` on the request, also stores
   ``model_id`` under ``openai_pc_aff:<sha256(body)>:prompt_caching`` (parallel to
   ``deployment:<hash>:prompt_caching``). The hashed ``body`` includes the Router
   ``model`` alias, ``user_api_key_hash``, and ``prompt_cache_key`` so different
   model groups do not share affinity entries. Requires the
   same minimum prompt size as :func:`litellm.utils.is_prompt_caching_valid_prompt`
   (default 1024 tokens). TTL is 300 seconds (same as ``PromptCachingCache`` writes).

Both behaviors are gated by Router ``optional_pre_call_checks: ["prompt_caching"]``
only (no separate flag). Lookup order: ``cache_control`` prefix hit first, then
OpenAI ``prompt_cache_key`` affinity.

**Tenant token**: ``user_api_key_hash`` from ``metadata`` / ``litellm_metadata`` (Proxy
virtual key identity). Empty string if absent (same key string may then be shared
across callers—prefer setting metadata in production).
"""

import hashlib
from typing import Any, List, Optional, cast

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import is_prompt_caching_valid_prompt

# Redis namespace for OpenAI prompt_cache_key affinity (distinct from PromptCachingCache keys).
# PromptCachingCache / PromptCachingCacheValue are imported inside methods to avoid a module-level
# cycle (litellm/__init__ -> Router -> this module -> prompt_caching_cache -> ...).
_OPENAI_PC_AFFINITY_KEY_PREFIX = "openai_pc_aff"


def _extract_prompt_cache_key(request_kwargs: Optional[dict]) -> Optional[str]:
    if not request_kwargs:
        return None
    raw = request_kwargs.get("prompt_cache_key")
    if raw is None and isinstance(request_kwargs.get("optional_params"), dict):
        raw = request_kwargs["optional_params"].get("prompt_cache_key")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _tenant_token_for_openai_pc_affinity(request_kwargs: Optional[dict]) -> str:
    """Stable tenant id for affinity keys: Proxy ``metadata.user_api_key_hash`` when present."""
    if not request_kwargs:
        return ""
    for key in ("litellm_metadata", "metadata"):
        md = request_kwargs.get(key)
        if isinstance(md, dict):
            h = md.get("user_api_key_hash")
            if h is not None:
                return str(h)
    return ""


def _tenant_token_from_standard_payload(payload: StandardLoggingPayload) -> str:
    meta = payload.get("metadata") or {}
    h = meta.get("user_api_key_hash")
    return str(h) if h is not None else ""


def _openai_prompt_cache_affinity_cache_key(
    router_model: str, tenant_token: str, prompt_cache_key: str
) -> str:
    body = f"{router_model}|{tenant_token}|{prompt_cache_key}"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{_OPENAI_PC_AFFINITY_KEY_PREFIX}:{digest}:prompt_caching"


def _parse_model_id_from_affinity_cache_value(cache_result: Any) -> Optional[str]:
    if cache_result is None:
        return None
    if isinstance(cache_result, dict):
        mid = cache_result.get("model_id")
        return str(mid) if mid is not None else None
    if isinstance(cache_result, str):
        return cache_result
    return None


def _model_supports_prompt_cache_key_param(
    model: str,
    custom_llm_provider: Optional[str],
) -> bool:
    """
    True when the provider's chat config lists ``prompt_cache_key`` (declared under
    ``litellm/llms/``, not hardcoded here).
    """
    # Import locally to avoid circular import: litellm/__init__ -> Router -> this module.
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,
    )

    try:
        params = get_supported_openai_params(
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
    except Exception:
        return False
    return bool(params and "prompt_cache_key" in params)


def _healthy_deployments_include_prompt_cache_key_support(
    healthy_deployments: List[dict],
) -> bool:
    for deployment in healthy_deployments:
        lp = deployment.get("litellm_params")
        if not isinstance(lp, dict):
            continue
        dep_model = lp.get("model")
        if dep_model is None:
            continue
        if _model_supports_prompt_cache_key_param(
            str(dep_model),
            lp.get("custom_llm_provider"),
        ):
            return True
    return False


def _should_apply_prompt_cache_key_affinity(
    model: str,
    request_kwargs: dict,
    healthy_deployments: List[dict],
) -> bool:
    if _model_supports_prompt_cache_key_param(
        model,
        request_kwargs.get("custom_llm_provider"),
    ):
        return True
    return _healthy_deployments_include_prompt_cache_key_support(healthy_deployments)


class PromptCachingDeploymentCheck(CustomLogger):
    def __init__(self, cache: DualCache):
        self.cache = cache

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        typed_healthy_deployments = cast(List[dict], healthy_deployments)
        request_kwargs = request_kwargs or {}

        if messages is None or not is_prompt_caching_valid_prompt(
            messages=messages,
            model=model,
        ):
            return typed_healthy_deployments

        from litellm.router_utils.prompt_caching_cache import PromptCachingCache

        prompt_cache = PromptCachingCache(
            cache=self.cache,
        )

        model_id_dict = await prompt_cache.async_get_model_id(
            messages=cast(List[AllMessageValues], messages),
            tools=None,
        )
        if model_id_dict is not None:
            model_id = model_id_dict["model_id"]
            for deployment in typed_healthy_deployments:
                if deployment["model_info"]["id"] == model_id:
                    return [deployment]

        prompt_cache_key = _extract_prompt_cache_key(request_kwargs)
        if prompt_cache_key is None:
            return typed_healthy_deployments

        if not _should_apply_prompt_cache_key_affinity(
            model=model,
            request_kwargs=request_kwargs,
            healthy_deployments=typed_healthy_deployments,
        ):
            return typed_healthy_deployments

        cache_key = _openai_prompt_cache_affinity_cache_key(
            router_model=model,
            tenant_token=_tenant_token_for_openai_pc_affinity(request_kwargs),
            prompt_cache_key=prompt_cache_key,
        )
        cache_result = await self.cache.async_get_cache(key=cache_key)
        pinned_id = _parse_model_id_from_affinity_cache_value(cache_result)
        if pinned_id is None:
            return typed_healthy_deployments

        for deployment in typed_healthy_deployments:
            if deployment["model_info"]["id"] == pinned_id:
                return [deployment]

        return typed_healthy_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )

        if standard_logging_object is None:
            return

        call_type = standard_logging_object["call_type"]

        if (
            call_type != CallTypes.completion.value
            and call_type != CallTypes.acompletion.value
            and call_type != CallTypes.anthropic_messages.value
        ):  # only use prompt caching for completion calls
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, CALL TYPE IS NOT COMPLETION or ANTHROPIC MESSAGE"
            )
            return

        model = standard_logging_object["model"]
        messages = standard_logging_object["messages"]
        model_id = standard_logging_object["model_id"]

        if messages is None or not isinstance(messages, list):
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MESSAGES IS NOT A LIST"
            )
            return
        if model_id is None:
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MODEL ID IS NONE"
            )
            return

        from litellm.router_utils.prompt_caching_cache import (
            PromptCachingCache,
            PromptCachingCacheValue,
        )

        ## PROMPT CACHING - cache model id, if prompt caching valid prompt + provider
        if is_prompt_caching_valid_prompt(
            model=model,
            messages=cast(List[AllMessageValues], messages),
        ):
            cache = PromptCachingCache(
                cache=self.cache,
            )
            await cache.async_add_model_id(
                model_id=model_id,
                messages=messages,
                tools=None,  # [TODO]: add tools once standard_logging_object supports it
            )

        prompt_cache_key = _extract_prompt_cache_key(kwargs)
        resolved_provider = standard_logging_object.get("custom_llm_provider")
        if resolved_provider is None:
            resolved_provider = kwargs.get("custom_llm_provider")

        if (
            prompt_cache_key is not None
            and _model_supports_prompt_cache_key_param(model, resolved_provider)
            and is_prompt_caching_valid_prompt(
                model=model,
                messages=cast(List[AllMessageValues], messages),
            )
        ):
            router_model = standard_logging_object.get("model_group") or model
            cache_key = _openai_prompt_cache_affinity_cache_key(
                router_model=str(router_model),
                tenant_token=_tenant_token_from_standard_payload(
                    standard_logging_object
                ),
                prompt_cache_key=prompt_cache_key,
            )
            try:
                await self.cache.async_set_cache(
                    cache_key,
                    PromptCachingCacheValue(model_id=str(model_id)),
                    ttl=300,
                )
            except Exception as e:
                verbose_logger.debug(
                    "PromptCachingDeploymentCheck: failed to set openai prompt_cache_key affinity cache router_model=%s error=%s",
                    router_model,
                    e,
                )

        return
