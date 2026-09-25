from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

from pydantic import BaseModel, JsonValue, TypeAdapter

import litellm
from litellm._internal_context import current_billing_time, pinned_billing_time
from litellm.caching.dual_cache import DualCache
from litellm.llms.anthropic.prompt_cache_prediction import (
    NativePredictionTarget,
    PromptPrefix,
    TokenCounter,
    UnsupportedPredictionTarget,
    cache_scope,
    count_prompt_tokens,
    parse_prompt,
    resolve_prediction_target,
    supported_prediction_headers,
)
from litellm.proxy.common_utils.prompt_cache_pricing import price_cache_tokens
from litellm.proxy.hooks.prompt_cache_prediction import lookup
from litellm.types.management_endpoints.prompt_cache_prediction import (
    CacheCostScenario,
    CacheEvidence,
    CachePredictionArm,
    CacheTokenBuckets,
)
from litellm.types.router import Deployment
from litellm.utils import get_prompt_cache_min_tokens

__all__: Final = ("AnthropicCacheRouting", "TokenCounter", "predict_arm")

_JSON: Final = TypeAdapter(Mapping[str, JsonValue])
_NATIVE_OPTIONS: Final = frozenset(
    (
        "max_tokens",
        "system",
        "tools",
        "tool_choice",
        "thinking",
        "output_config",
        "cache_control",
        "speed",
        "service_tier",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
    )
)


class _ModelLimits(BaseModel):
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AnthropicCacheRouting:
    body: Mapping[str, JsonValue]
    prefix: PromptPrefix
    requested_output_limit: int

    @staticmethod
    def request_body(
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, JsonValue],
        request_kwargs: Mapping[str, object],
        messages: Sequence[Mapping[str, object]] | None,
    ) -> Mapping[str, JsonValue] | None:
        if not urlparse(url).path.endswith("/v1/messages") or not supported_prediction_headers(headers):
            return None
        return _JSON.validate_python(
            MappingProxyType(
                {
                    **body,
                    **MappingProxyType({key: request_kwargs[key] for key in _NATIVE_OPTIONS if key in request_kwargs}),
                    "messages": messages,
                }
            )
        )

    @classmethod
    def from_body(cls, body: Mapping[str, JsonValue]) -> AnthropicCacheRouting | None:
        prefix: Final = parse_prompt(body)
        limit: Final = body.get("max_tokens")
        if prefix is None or not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return None
        return cls(body, prefix, limit)

    @staticmethod
    def supports(deployment: Deployment) -> bool:
        return isinstance(resolve_prediction_target(deployment.litellm_params), NativePredictionTarget)

    async def is_warm(self, deployment: Deployment, caller: str, cache: DualCache, now: float) -> bool:
        target: Final = resolve_prediction_target(deployment.litellm_params)
        if not isinstance(target, NativePredictionTarget):
            return False
        scope: Final = cache_scope(caller, deployment.model_info.id or "", target.api_key, target.model)
        observation: Final = await lookup(cache, scope, self.prefix, now=now)
        return observation is not None and observation.expires_at > now

    @staticmethod
    def fits(deployment: Deployment, input_tokens: int, output_tokens: int) -> bool:
        target: Final = resolve_prediction_target(deployment.litellm_params)
        if not isinstance(target, NativePredictionTarget):
            return False
        limits: Final = _ModelLimits.model_validate(
            MappingProxyType(
                {
                    **litellm.get_model_info(target.model, custom_llm_provider="anthropic"),
                    **deployment.model_info.model_dump(exclude_none=True),
                }
            )
        )
        return (
            limits.max_input_tokens is not None
            and input_tokens + output_tokens <= limits.max_input_tokens
            and limits.max_output_tokens is not None
            and output_tokens <= limits.max_output_tokens
        )

    async def predict(
        self,
        deployment: Deployment,
        caller: str,
        cache: DualCache,
        counter: TokenCounter,
        now: float | None,
    ) -> CachePredictionArm:
        return await predict_arm(deployment, self.body, self.prefix, caller, cache, counter, now=now)

    @staticmethod
    def cost(arm: CachePredictionArm, output_tokens: int) -> float | None:
        return (
            price_cache_tokens(arm.model or "", arm.deployment_id, arm.estimate.tokens, output_tokens)
            if arm.estimate is not None
            else None
        )

    @staticmethod
    async def count_tokens(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        return await count_prompt_tokens(model, api_key, body)


def _buckets(prefix_tokens: int, suffix_tokens: int, read_tokens: int, ttl_seconds: int) -> CacheTokenBuckets:
    return CacheTokenBuckets(
        uncached_input_tokens=suffix_tokens,
        cache_read_input_tokens=read_tokens,
        cache_creation_5m_input_tokens=prefix_tokens - read_tokens if ttl_seconds == 300 else 0,
        cache_creation_1h_input_tokens=prefix_tokens - read_tokens if ttl_seconds == 3600 else 0,
    )


def _scenario(model: str, deployment_id: str, tokens: CacheTokenBuckets) -> CacheCostScenario | None:
    cost: Final = price_cache_tokens(model=model, deployment_id=deployment_id, tokens=tokens)
    return CacheCostScenario(tokens=tokens, input_cost=cost) if cost is not None else None


async def predict_arm(
    deployment: Deployment,
    body: Mapping[str, JsonValue],
    prefix: PromptPrefix,
    caller_key_hash: str,
    cache: DualCache,
    token_counter: TokenCounter,
    now: float | None = None,
) -> CachePredictionArm:
    deployment_id: Final = deployment.model_info.id or ""
    params: Final = deployment.litellm_params
    unknown: Final = CachePredictionArm(deployment_id=deployment_id, model=params.model)
    if deployment.model_info.blocked:
        return unknown.model_copy(update=MappingProxyType({"reason": "unsupported_deployment_configuration"}))
    target: Final = resolve_prediction_target(params)
    if isinstance(target, UnsupportedPredictionTarget):
        return unknown.model_copy(update=MappingProxyType({"reason": target.reason}))
    model: Final = target.model
    api_key: Final = target.api_key
    total_count: Final = await token_counter(model, api_key, body)
    prefix_count: Final = await token_counter(model, api_key, prefix.prefix_body)
    if total_count is None or prefix_count is None or total_count < prefix_count:
        return unknown.model_copy(update=MappingProxyType({"reason": "token_count_unavailable"}))
    scope: Final = cache_scope(caller_key_hash, deployment_id, api_key, model)
    checked_at: Final = time.time() if now is None else now
    observation: Final = await lookup(cache, scope, prefix, now=checked_at)
    exact: Final = observation is not None and observation.fingerprint == prefix.fingerprint
    cacheable: Final = observation.cached_tokens if exact and observation is not None else prefix_count
    if cacheable > total_count or (observation is not None and observation.cached_tokens > cacheable):
        return unknown.model_copy(update=MappingProxyType({"reason": "inconsistent_prefix_token_count"}))
    suffix: Final = total_count - cacheable
    evidence: Final = (
        CacheEvidence(observed_at=observation.observed_at, expires_at=observation.expires_at)
        if observation is not None
        else None
    )
    if cacheable < get_prompt_cache_min_tokens(params.model):
        disabled: Final = _scenario(model, deployment_id, CacheTokenBuckets(uncached_input_tokens=total_count))
        if disabled is None:
            return unknown.model_copy(update=MappingProxyType({"reason": "pricing_unavailable"}))
        return CachePredictionArm(
            deployment_id=deployment_id,
            model=model,
            cache_state="disabled",
            reason="below_cache_minimum",
            estimate=disabled,
            cold=disabled,
            warm=disabled,
            token_count_source="anthropic_count_tokens",
        )
    fresh: Final = observation is not None and observation.expires_at > checked_at
    read: Final = observation.cached_tokens if fresh and observation is not None else 0
    with pinned_billing_time(current_billing_time()):
        cold: Final = _scenario(model, deployment_id, _buckets(cacheable, suffix, 0, prefix.ttl_seconds))
        warm: Final = _scenario(model, deployment_id, _buckets(cacheable, suffix, cacheable, prefix.ttl_seconds))
        estimate: Final = _scenario(model, deployment_id, _buckets(cacheable, suffix, read, prefix.ttl_seconds))
    if cold is None or warm is None or estimate is None:
        return unknown.model_copy(update=MappingProxyType({"reason": "pricing_unavailable"}))
    return CachePredictionArm(
        deployment_id=deployment_id,
        model=model,
        cache_state="warm" if fresh and exact else "partial" if fresh else "stale" if observation else "unknown",
        reason=None if fresh else "observation_expired" if observation else "no_compatible_observation",
        estimate=estimate,
        cold=cold,
        warm=warm,
        evidence=evidence,
        token_count_source="anthropic_count_tokens",
    )
