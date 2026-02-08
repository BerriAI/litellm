"""Core smart routing logic for auto-routing.

Combines message classification and tier mapping to produce a
RouteDecision for each incoming request. No external dependencies.

Ported from ClawRouter's smart_router with Python 3.9 compatibility.
Backend/credential management removed — litellm handles that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from litellm._logging import verbose_router_logger

from .classifier import (
    ClassificationResult,
    ClassificationRule,
    TaskCategory,
    classify_task,
)
from .tiers import ModelTier, RoutingConfig


@dataclass(frozen=True)
class RouteDecision:
    """The result of routing a request: which model to use."""

    model: str  # e.g. "anthropic/claude-sonnet-4-5"
    tier: ModelTier
    category: TaskCategory
    confidence: float
    original_model: str  # The model requested by the client


class SmartRouter:
    """Routes messages to the optimal model based on classification."""

    def __init__(
        self,
        routing_config: Optional[RoutingConfig] = None,
        default_tier: ModelTier = ModelTier.MID,
    ):
        self.config = routing_config or RoutingConfig()
        self.default_tier = default_tier

        # Build custom classification rules from config
        self._custom_rules = self._build_custom_rules()

    def _build_custom_rules(self) -> List[ClassificationRule]:
        """Build ClassificationRule objects from YAML custom_patterns."""
        rules: List[ClassificationRule] = []
        for entry in self.config.custom_patterns:
            if not isinstance(entry, dict):
                continue
            pattern_str = entry.get("pattern")
            category_str = entry.get("category")
            if not pattern_str or not category_str:
                continue
            try:
                category = TaskCategory(category_str)
                compiled = re.compile(pattern_str, re.IGNORECASE)
                rules.append(
                    ClassificationRule(
                        pattern=compiled,
                        category=category,
                        confidence=0.90,
                        description="custom: %s" % pattern_str,
                    )
                )
            except (ValueError, re.error) as e:
                verbose_router_logger.warning(
                    "Skipping invalid custom pattern %s: %s",
                    pattern_str,
                    str(e),
                )
        return rules

    # Known bare model name -> canonical "provider/model" mappings.
    _MODEL_ALIASES: Dict[str, str] = {
        # Claude 4.x family
        "claude-opus-4-6": "anthropic/claude-opus-4",
        "claude-opus-4": "anthropic/claude-opus-4",
        "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
        "claude-sonnet-4-5-20250514": "anthropic/claude-sonnet-4-5",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "claude-haiku-4-5": "anthropic/claude-haiku-4",
        "claude-haiku-4": "anthropic/claude-haiku-4",
        # Claude 3.x family
        "claude-3-opus-20240229": "anthropic/claude-opus-4",
        "claude-3-5-sonnet-20241022": "anthropic/claude-sonnet-4-5",
        "claude-3-5-haiku-20241022": "anthropic/claude-haiku-4",
    }

    def _normalize_model(self, model: str) -> Optional[str]:
        """Normalize a bare model name to its canonical provider/model form.

        Returns the canonical ID if found, or None if the model is unknown.
        """
        # Already has a provider prefix
        if "/" in model:
            return model

        # Exact alias match
        canonical = self._MODEL_ALIASES.get(model)
        if canonical:
            return canonical

        # Heuristic: claude-* -> strip date suffix -> try alias
        if model.startswith("claude-"):
            # Strip date suffix like -20250514
            stripped = re.sub(r"-\d{8}$", "", model)
            canonical = self._MODEL_ALIASES.get(stripped)
            if canonical:
                return canonical

            # Fallback: prefix with anthropic/
            return "anthropic/%s" % stripped

        return None

    def resolve_route(
        self,
        messages: List[Dict[str, Any]],
        requested_model: str = "auto",
    ) -> RouteDecision:
        """Resolve the optimal route for a set of messages.

        Args:
            messages: OpenAI-format messages list.
            requested_model: The model field from the request. If "auto",
                smart routing is applied. Otherwise, the model is used as-is.

        Returns:
            RouteDecision with the selected model, tier, etc.
        """
        # Extract the last user message for classification
        user_text = self._extract_user_text(messages)

        # If a specific model was requested (not "auto"), use it directly
        if requested_model != "auto":
            if self._is_tier_model(requested_model):
                return self._route_specific_model(requested_model, user_text)

            # Try to normalize the model name
            normalized = self._normalize_model(requested_model)
            if normalized and self._is_tier_model(normalized):
                return self._route_specific_model(normalized, user_text)

            verbose_router_logger.info(
                "Unknown model requested, using smart routing: %s",
                requested_model,
            )

        # Classify the message
        classification = classify_task(user_text, self._custom_rules or None)

        verbose_router_logger.debug(
            "Message classified: category=%s, confidence=%.2f, pattern=%s, text_preview=%s",
            classification.category,
            classification.confidence,
            classification.matched_pattern,
            user_text[:80] if user_text else "",
        )

        # Map category -> tier
        tier = self.config.get_tier_for_category(classification.category)

        return self._route_to_tier(
            tier,
            classification.category,
            classification.confidence,
            requested_model,
        )

    def _route_specific_model(
        self, model: str, user_text: str
    ) -> RouteDecision:
        """Route a request for a specific model (not "auto")."""
        # Still classify for logging/headers, but use the requested model
        classification = classify_task(user_text, self._custom_rules or None)

        # Determine tier from model (rough heuristic)
        tier = self._infer_tier_from_model(model)

        return RouteDecision(
            model=model,
            tier=tier,
            category=classification.category,
            confidence=classification.confidence,
            original_model=model,
        )

    def _route_to_tier(
        self,
        tier: ModelTier,
        category: TaskCategory,
        confidence: float,
        original_model: str,
    ) -> RouteDecision:
        """Route to a specific tier, selecting the configured model."""
        tier_config = self.config.get_tier_config(tier)
        model = tier_config.model

        return RouteDecision(
            model=model,
            tier=tier,
            category=category,
            confidence=confidence,
            original_model=original_model,
        )

    def _extract_user_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the last user message text for classification."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                # Handle content as list of content parts
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts)
        return ""

    def _is_tier_model(self, model: str) -> bool:
        """Check if *model* is one of the configured tier or fallback models."""
        for tier in ModelTier:
            tc = self.config.get_tier_config(tier)
            if model == tc.model:
                return True
            if model in tc.fallback_models:
                return True
        return False

    def _infer_tier_from_model(self, model: str) -> ModelTier:
        """Infer which tier a model belongs to based on its name."""
        model_lower = model.lower()

        # Top tier models
        if any(
            kw in model_lower
            for kw in [
                "opus",
                "sonnet-4",
                "o1",
                "o3",
                "gpt-4o",
                "gpt-5",
                "codex",
                "gemini-2.5-pro",
            ]
        ):
            return ModelTier.TOP

        # Mid tier models
        if any(kw in model_lower for kw in ["glm"]):
            return ModelTier.MID

        # Low tier models
        if any(
            kw in model_lower
            for kw in ["deepseek", "mini", "flash", "haiku", "nano", "small"]
        ):
            return ModelTier.LOW

        # Default to mid
        return ModelTier.MID
