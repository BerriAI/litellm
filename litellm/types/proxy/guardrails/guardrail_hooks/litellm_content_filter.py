from typing import List, Optional

from pydantic import Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class ProximityDetectionConfig(GuardrailConfigModel):
    """Configuration for proximity-based detection of harmful content."""

    max_distance: int = Field(
        default=5,
        description="Maximum distance (in tokens) between identity keywords and negative modifiers",
    )
    severity: str = Field(
        default="high",
        description="Severity level when proximity match is detected (high, medium, low)",
    )


class LitellmContentFilterGuardrailConfigModel(GuardrailConfigModel):
    """
    Configuration model for LiteLLM Content Filter guardrail.

    Supports:
    - Traditional keyword and pattern matching
    - Category-based detection (harmful content, bias detection)
    - Proximity-based detection (identity keywords + negative modifiers)
    """

    # Traditional patterns and keywords
    patterns: Optional[List[dict]] = Field(
        default=None,
        description="List of regex patterns to detect (prebuilt or custom)",
    )
    blocked_words: Optional[List[dict]] = Field(
        default=None,
        description="List of blocked keywords with actions",
    )
    blocked_words_file: Optional[str] = Field(
        default=None,
        description="Path to YAML file containing blocked words",
    )

    # Category-based detection
    categories: Optional[List[dict]] = Field(
        default=None,
        description="List of prebuilt categories to enable (harmful_*, bias_*)",
    )
    severity_threshold: str = Field(
        default="medium",
        description="Minimum severity to block (high, medium, low)",
    )

    # Proximity-based detection for bias
    identity_keywords: Optional[List[str]] = Field(
        default=None,
        description="List of identity keywords (e.g., 'gay', 'transgender', 'muslim') that are neutral by themselves",
    )
    negative_modifiers: Optional[List[str]] = Field(
        default=None,
        description="List of negative/harmful modifiers (e.g., 'unnatural', 'disease', 'sin')",
    )
    harmful_actions: Optional[List[str]] = Field(
        default=None,
        description="List of harmful action verbs (e.g., 'cure', 'fix', 'eliminate')",
    )

    # Proximity detection configuration
    identity_plus_negative: Optional[ProximityDetectionConfig] = Field(
        default=None,
        description="Config for detecting identity keywords near negative modifiers",
    )
    action_plus_identity: Optional[ProximityDetectionConfig] = Field(
        default=None,
        description="Config for detecting harmful actions near identity keywords",
    )

    # Redaction customization
    pattern_redaction_format: Optional[str] = Field(
        default="[{pattern_name}_REDACTED]",
        description="Format string for pattern redaction (use {pattern_name} placeholder)",
    )
    keyword_redaction_tag: Optional[str] = Field(
        default="[KEYWORD_REDACTED]",
        description="Tag to use for keyword redaction",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "LiteLLM Content Filter"
