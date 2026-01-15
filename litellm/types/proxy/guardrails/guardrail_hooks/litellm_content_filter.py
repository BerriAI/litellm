from enum import Enum
from typing import List, Literal, Optional, TypedDict, Union

from pydantic import Field

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


# Detection type enum
class DetectionType(str, Enum):
    PATTERN = "pattern"
    BLOCKED_WORD = "blocked_word"
    CATEGORY_KEYWORD = "category_keyword"


# Typed detection dictionaries
class PatternDetection(TypedDict):
    type: Literal["pattern"]
    pattern_name: str
    # Note: matched_text is intentionally excluded to avoid logging sensitive content
    action: str  # ContentFilterAction.value


class BlockedWordDetection(TypedDict):
    type: Literal["blocked_word"]
    keyword: str
    action: str  # ContentFilterAction.value
    description: Optional[str]


class CategoryKeywordDetection(TypedDict):
    type: Literal["category_keyword"]
    category: str
    keyword: str
    severity: str
    action: str  # ContentFilterAction.value


ContentFilterDetection = Union[PatternDetection, BlockedWordDetection, CategoryKeywordDetection]


class ContentFilterCategoryConfig(BaseLiteLLMOpenAIResponseObject):
    """
    category: "harmful_self_harm"
                  enabled: true
                  action: "BLOCK"
                  severity_threshold: "medium"
                  category_file: "/path/to/custom_file.yaml"  # optional override
    """

    category: str = Field(
        description="The category to detect",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the category is enabled",
    )
    action: Literal["BLOCK", "MASK"] = Field(
        description="The action to take when the category is detected",
    )
    severity_threshold: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="The severity threshold to detect the category",
    )
    category_file: Optional[str] = Field(
        default=None,
        description="Optional override. Use your own category file instead of the default one.",
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
    categories: Optional[List[ContentFilterCategoryConfig]] = Field(
        default=None,
        description="List of prebuilt categories to enable (harmful_*, bias_*)",
    )
    severity_threshold: str = Field(
        default="medium",
        description="Minimum severity to block (high, medium, low)",
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
