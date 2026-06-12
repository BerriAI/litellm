"""LLM-based prompt complexity classifier router.

Uses a small LLM (e.g. ollama/qwen2.5:0.5b) to classify prompt complexity and
route to a pre-configured tier model. Falls back to a rule-based
ComplexityRouter on timeout or error.
"""

import asyncio
import hashlib
import re
from time import monotonic
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

from .config import LLMClassifierRouterConfig
from .prompt_templates import TWO_TIER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class _TierCache:
    """Simple TTL cache. No LRU eviction — on overflow, clear and keep going."""

    def __init__(self, ttl_seconds: float, max_size: int) -> None:
        self._store: Dict[str, Tuple[str, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def get(self, key: str) -> Optional[str]:
        item = self._store.get(key)
        if item is None:
            return None
        tier, expires_at = item
        if monotonic() > expires_at:
            del self._store[key]
            return None
        return tier

    def set(self, key: str, tier: str) -> None:
        if len(self._store) >= self._max_size:
            self._store.clear()
        self._store[key] = (tier, monotonic() + self._ttl)


def _parse_tier(raw: str, valid_tiers: Tuple[str, ...]) -> str:
    """Parse a tier string from the classifier LLM's raw output.

    Strategy: exact fullmatch first (avoids "a SIMPLE example" false positive),
    then substring containment as a fallback for outputs like "SIMPLE.".
    """
    cleaned = raw.strip().upper()
    if valid_tiers:
        if re.fullmatch(r"|".join(re.escape(t) for t in valid_tiers), cleaned):
            return cleaned
    for tier in valid_tiers:
        if tier in cleaned:
            return tier
    raise ValueError(f"Could not parse tier from LLM output: {raw!r}")


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class LLMClassifierRouter(CustomLogger):
    """LLM-based prompt complexity classifier router.

    Classifies requests via a small LLM and routes them to one of the
    configured tier models. Falls back to a rule-based ComplexityRouter
    when the LLM call fails.
    """

    def __init__(
        self,
        model_name: str,
        litellm_router_instance: "Router",
        llm_classifier_router_config: Optional[Dict[str, Any]] = None,
        default_model: Optional[str] = None,
    ):
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance

        if llm_classifier_router_config:
            self.config = LLMClassifierRouterConfig(**llm_classifier_router_config)
        else:
            self.config = LLMClassifierRouterConfig()

        self._cache = _TierCache(
            ttl_seconds=self.config.cache_ttl_seconds,
            max_size=self.config.cache_max_size,
        )

        self._fallback_scorer: Optional[Any] = None
        if self.config.fallback_to_complexity_router:
            from litellm.router_strategy.complexity_router.complexity_router import (
                ComplexityRouter,
            )

            self._fallback_scorer = ComplexityRouter(
                model_name=f"{model_name}::fallback_scorer",
                litellm_router_instance=litellm_router_instance,
            )

        verbose_router_logger.debug(f"LLMClassifierRouter initialized for {model_name} with tiers: {self.config.tiers}")

    def _get_system_prompt(self) -> str:
        return self.config.classifier_system_prompt or TWO_TIER_SYSTEM_PROMPT

    async def _call_classifier_llm(self, prompt: str) -> str:
        """Call the classifier LLM directly via litellm.acompletion.

        Bypasses the Router instance to avoid recursion. The litellm
        import is lazy so this module can be imported during Router init
        without circular-import issues.
        """
        import litellm

        truncated = prompt[: self.config.classifier_max_input_chars]
        response = await litellm.acompletion(
            model=self.config.classifier_model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": truncated},
            ],
            temperature=self.config.classifier_temperature,
            max_tokens=self.config.classifier_max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    async def classify(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Classify a prompt and return (tier, method).

        method ∈ {"cache", "llm", "fallback_rule", "fallback_default"}
        """
        valid_tiers = tuple(self.config.tiers.keys())

        if self.config.enable_cache:
            cached = self._cache.get(_cache_key(prompt))
            if cached is not None and cached in valid_tiers:
                return cached, "cache"

        try:
            raw = await asyncio.wait_for(
                self._call_classifier_llm(prompt),
                timeout=self.config.classifier_timeout,
            )
            tier = _parse_tier(raw, valid_tiers)
            if self.config.enable_cache:
                self._cache.set(_cache_key(prompt), tier)
            return tier, "llm"
        except Exception as e:
            verbose_router_logger.warning(
                f"LLMClassifierRouter: LLM classification failed ({type(e).__name__}: {e}), falling back"
            )

        if self._fallback_scorer is not None:
            try:
                complexity_tier, _, _ = self._fallback_scorer.classify(prompt, system_prompt)
                tier_name = complexity_tier.value
                if tier_name in valid_tiers:
                    return tier_name, "fallback_rule"
            except Exception as e:
                verbose_router_logger.warning(f"LLMClassifierRouter: fallback scorer failed ({e})")

        fallback = self.config.fallback_tier
        if fallback not in valid_tiers and valid_tiers:
            fallback = valid_tiers[0]
        return fallback, "fallback_default"

    def get_model_for_tier(self, tier: str) -> Optional[str]:
        return self.config.tiers.get(tier)

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """Pre-routing hook that classifies the request and selects a tier model."""
        from litellm.types.router import PreRoutingHookResponse

        if self._fallback_scorer is None:
            from litellm.router_strategy.complexity_router.complexity_router import (
                ComplexityRouter,
            )

            helper = ComplexityRouter(
                model_name=f"{self.model_name}::hook_helper",
                litellm_router_instance=self.litellm_router_instance,
            )
        else:
            helper = self._fallback_scorer

        resolved_messages = helper._resolve_messages(messages, request_kwargs)
        if not resolved_messages:
            verbose_router_logger.debug("LLMClassifierRouter: No messages could be resolved, skipping routing")
            return None

        has_original_messages = messages is not None and len(messages) > 0
        user_message, system_prompt = helper._extract_user_message_and_system_prompt(resolved_messages)

        if user_message is None:
            verbose_router_logger.debug("LLMClassifierRouter: No user message found, skipping routing")
            return None

        tier, method = await self.classify(user_message, system_prompt)
        routed_model = self.get_model_for_tier(tier) or self.get_model_for_tier(self.config.fallback_tier)

        verbose_router_logger.info(f"LLMClassifierRouter: tier={tier}, method={method}, routed_model={routed_model}")

        metadata = request_kwargs.setdefault("metadata", {})
        metadata["llm_classifier_router_tier"] = tier
        metadata["llm_classifier_router_method"] = method
        metadata["llm_classifier_router_model"] = routed_model

        return PreRoutingHookResponse(
            model=routed_model,
            messages=messages if has_original_messages else None,
        )
