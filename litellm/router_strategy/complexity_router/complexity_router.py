"""
Complexity-based Auto Router

A rule-based routing strategy that uses weighted scoring across multiple dimensions
to classify requests by complexity and route them to appropriate models.

No external API calls - all scoring is local and <1ms.

Inspired by ClawRouter: https://github.com/BlockRunAI/ClawRouter
"""
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

from .config import (
    DEFAULT_CODE_KEYWORDS,
    DEFAULT_REASONING_KEYWORDS,
    DEFAULT_SIMPLE_KEYWORDS,
    DEFAULT_TECHNICAL_KEYWORDS,
    ComplexityRouterConfig,
    ComplexityTier,
)

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import PreRoutingHookResponse
else:
    Router = Any
    PreRoutingHookResponse = Any


class DimensionScore:
    """Represents a score for a single dimension with optional signal."""
    
    __slots__ = ("name", "score", "signal")
    
    def __init__(self, name: str, score: float, signal: Optional[str] = None):
        self.name = name
        self.score = score
        self.signal = signal


class ComplexityRouter(CustomLogger):
    """
    Rule-based complexity router that classifies requests and routes to appropriate models.
    
    Handles requests in <1ms with zero external API calls by using weighted scoring
    across multiple dimensions:
    - Token count (short=simple, long=complex)
    - Code presence (code keywords → complex)
    - Reasoning markers ("step by step", "think through" → reasoning tier)
    - Technical terms (domain complexity)
    - Simple indicators ("what is", "define" → simple, negative weight)
    - Multi-step patterns ("first...then", numbered steps)
    - Question complexity (multiple questions)
    """
    
    def __init__(
        self,
        model_name: str,
        litellm_router_instance: "Router",
        complexity_router_config: Optional[Dict[str, Any]] = None,
        default_model: Optional[str] = None,
    ):
        """
        Initialize ComplexityRouter.
        
        Args:
            model_name: The name of the model/deployment using this router.
            litellm_router_instance: The LiteLLM Router instance.
            complexity_router_config: Optional configuration dict from proxy config.
            default_model: Optional default model to use if tier cannot be determined.
        """
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance
        
        # Parse config - always create a new instance to avoid singleton mutation
        if complexity_router_config:
            self.config = ComplexityRouterConfig(**complexity_router_config)
        else:
            self.config = ComplexityRouterConfig()
        
        # Override default_model if provided
        if default_model:
            self.config.default_model = default_model
        
        # Build effective keyword lists (use config overrides or defaults)
        self.code_keywords = self.config.code_keywords or DEFAULT_CODE_KEYWORDS
        self.reasoning_keywords = self.config.reasoning_keywords or DEFAULT_REASONING_KEYWORDS
        self.technical_keywords = self.config.technical_keywords or DEFAULT_TECHNICAL_KEYWORDS
        self.simple_keywords = self.config.simple_keywords or DEFAULT_SIMPLE_KEYWORDS
        
        # Pre-compile regex patterns for efficiency
        # Use non-greedy .*? to prevent ReDoS on pathological inputs
        self._multi_step_patterns = [
            re.compile(r"first.*?then", re.IGNORECASE),
            re.compile(r"step\s*\d", re.IGNORECASE),
            re.compile(r"\d+\.\s"),
            re.compile(r"[a-z]\)\s", re.IGNORECASE),
        ]
        
        verbose_router_logger.debug(
            f"ComplexityRouter initialized for {model_name} with tiers: {self.config.tiers}"
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        Uses a simple heuristic: ~4 characters per token on average.
        """
        return len(text) // 4
    
    def _score_token_count(self, estimated_tokens: int) -> DimensionScore:
        """Score based on token count."""
        thresholds = self.config.token_thresholds
        simple_threshold = thresholds.get("simple", 15)
        complex_threshold = thresholds.get("complex", 400)
        
        if estimated_tokens < simple_threshold:
            return DimensionScore(
                "tokenCount", 
                -1.0, 
                f"short ({estimated_tokens} tokens)"
            )
        if estimated_tokens > complex_threshold:
            return DimensionScore(
                "tokenCount", 
                1.0, 
                f"long ({estimated_tokens} tokens)"
            )
        return DimensionScore("tokenCount", 0, None)
    
    def _keyword_matches(self, text: str, keyword: str) -> bool:
        """
        Check if a keyword matches in text using word boundary matching.

        For single-word keywords, uses regex word boundaries to avoid
        false positives (e.g., "error" matching "terrorism", "class" matching "classical").
        For multi-word phrases, uses substring matching.
        """
        kw_lower = keyword.lower()

        # For single-word keywords, use word boundary matching to avoid false positives
        # e.g., "api" should not match "capital", "error" should not match "terrorism"
        if " " not in kw_lower:
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            return bool(re.search(pattern, text))

        # For multi-word phrases, substring matching is fine
        return kw_lower in text
    
    def _score_keyword_match(
        self,
        text: str,
        keywords: List[str],
        name: str,
        signal_label: str,
        thresholds: Tuple[int, int],  # (low, high)
        scores: Tuple[float, float, float],  # (none, low, high)
    ) -> Tuple[DimensionScore, int]:
        """Score based on keyword matches using word boundary matching.

        Returns:
            Tuple of (DimensionScore, match_count) so callers can reuse the count.
        """
        low_threshold, high_threshold = thresholds
        score_none, score_low, score_high = scores

        matches = [kw for kw in keywords if self._keyword_matches(text, kw)]
        match_count = len(matches)

        if match_count >= high_threshold:
            return DimensionScore(
                name,
                score_high,
                f"{signal_label} ({', '.join(matches[:3])})"
            ), match_count
        if match_count >= low_threshold:
            return DimensionScore(
                name,
                score_low,
                f"{signal_label} ({', '.join(matches[:3])})"
            ), match_count
        return DimensionScore(name, score_none, None), match_count
    
    def _score_multi_step(self, text: str) -> DimensionScore:
        """Score based on multi-step patterns."""
        hits = sum(1 for p in self._multi_step_patterns if p.search(text))
        if hits > 0:
            return DimensionScore("multiStepPatterns", 0.5, "multi-step")
        return DimensionScore("multiStepPatterns", 0, None)
    
    def _score_question_complexity(self, text: str) -> DimensionScore:
        """Score based on number of question marks."""
        count = text.count("?")
        if count > 3:
            return DimensionScore(
                "questionComplexity", 
                0.5, 
                f"{count} questions"
            )
        return DimensionScore("questionComplexity", 0, None)
    
    def classify(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Tuple[ComplexityTier, float, List[str]]:
        """
        Classify a prompt by complexity.
        
        Args:
            prompt: The user's prompt/message.
            system_prompt: Optional system prompt for context.
            
        Returns:
            Tuple of (tier, score, signals) where:
            - tier: The ComplexityTier (SIMPLE, MEDIUM, COMPLEX, REASONING)
            - score: The raw weighted score
            - signals: List of triggered signals for debugging
        """
        # Combine text for analysis.
        # System prompt is intentionally included in code/technical/simple scoring
        # because it provides deployment-level context (e.g., "You are a Python assistant"
        # signals that code-capable models are appropriate). Reasoning markers use
        # user_text only to prevent system prompts from forcing REASONING tier.
        full_text = f"{system_prompt or ''} {prompt}".lower()
        user_text = prompt.lower()
        
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(prompt)
        
        # Score all dimensions, capturing match counts where needed
        code_score, _ = self._score_keyword_match(
            full_text, self.code_keywords, "codePresence", "code",
            (1, 2), (0, 0.5, 1.0),
        )
        reasoning_score, reasoning_match_count = self._score_keyword_match(
            user_text, self.reasoning_keywords, "reasoningMarkers", "reasoning",
            (1, 2), (0, 0.7, 1.0),
        )
        technical_score, _ = self._score_keyword_match(
            full_text, self.technical_keywords, "technicalTerms", "technical",
            (2, 4), (0, 0.5, 1.0),
        )
        simple_score, _ = self._score_keyword_match(
            full_text, self.simple_keywords, "simpleIndicators", "simple",
            (1, 2), (0, -1.0, -1.0),
        )

        dimensions: List[DimensionScore] = [
            self._score_token_count(estimated_tokens),
            code_score,
            reasoning_score,
            technical_score,
            simple_score,
            self._score_multi_step(full_text),
            self._score_question_complexity(prompt),
        ]

        # Collect signals
        signals = [d.signal for d in dimensions if d.signal is not None]

        # Compute weighted score
        weights = self.config.dimension_weights
        weighted_score = sum(
            d.score * weights.get(d.name, 0)
            for d in dimensions
        )

        # Check for reasoning override (2+ reasoning markers)
        # Reuse match count from _score_keyword_match to avoid scanning twice
        if reasoning_match_count >= 2:
            return ComplexityTier.REASONING, weighted_score, signals
        
        # Map score to tier
        boundaries = self.config.tier_boundaries
        simple_medium = boundaries.get("simple_medium", 0.15)
        medium_complex = boundaries.get("medium_complex", 0.35)
        complex_reasoning = boundaries.get("complex_reasoning", 0.60)
        
        if weighted_score < simple_medium:
            tier = ComplexityTier.SIMPLE
        elif weighted_score < medium_complex:
            tier = ComplexityTier.MEDIUM
        elif weighted_score < complex_reasoning:
            tier = ComplexityTier.COMPLEX
        else:
            tier = ComplexityTier.REASONING
        
        return tier, weighted_score, signals
    
    def get_model_for_tier(self, tier: ComplexityTier) -> str:
        """
        Get the model name for a given complexity tier.
        
        Args:
            tier: The complexity tier.
            
        Returns:
            The model name configured for that tier.
        """
        tier_key = tier.value if isinstance(tier, ComplexityTier) else tier
        
        # Check config tiers mapping
        model = self.config.tiers.get(tier_key)
        if model:
            return model
        
        # Fallback to default model if configured
        if self.config.default_model:
            return self.config.default_model
        
        # Last resort: return MEDIUM tier model or error
        medium_model = self.config.tiers.get(ComplexityTier.MEDIUM.value)
        if medium_model:
            return medium_model
        
        raise ValueError(
            f"No model configured for tier {tier_key} and no default_model set"
        )
    
    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: Dict,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
    ) -> Optional["PreRoutingHookResponse"]:
        """
        Pre-routing hook called before the routing decision.
        
        Classifies the request by complexity and returns the appropriate model.
        
        Args:
            model: The original model name requested.
            request_kwargs: The request kwargs.
            messages: The messages in the request.
            input: Optional input for embeddings.
            specific_deployment: Whether a specific deployment was requested.
            
        Returns:
            PreRoutingHookResponse with the routed model, or None if no routing needed.
        """
        from litellm.types.router import PreRoutingHookResponse
        
        if messages is None or len(messages) == 0:
            verbose_router_logger.debug(
                "ComplexityRouter: No messages provided, skipping routing"
            )
            return None
        
        # Extract the last user message and the last system prompt
        user_message: Optional[str] = None
        system_prompt: Optional[str] = None
        
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                if role == "user" and user_message is None:
                    user_message = content
                elif role == "system" and system_prompt is None:
                    system_prompt = content
        
        if user_message is None:
            verbose_router_logger.debug(
                "ComplexityRouter: No user message found, skipping routing"
            )
            return None
        
        # Classify the request
        tier, score, signals = self.classify(user_message, system_prompt)
        
        # Get the model for this tier
        routed_model = self.get_model_for_tier(tier)
        
        verbose_router_logger.info(
            f"ComplexityRouter: tier={tier.value}, score={score:.3f}, "
            f"signals={signals}, routed_model={routed_model}"
        )
        
        return PreRoutingHookResponse(
            model=routed_model,
            messages=messages,
        )
