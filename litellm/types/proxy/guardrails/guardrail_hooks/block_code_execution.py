"""Types for the Block Code Execution guardrail."""

from typing import Any, List, Literal, Optional, TypedDict, cast

from pydantic import Field

from .base import GuardrailConfigModel

CodeBlockActionTaken = Literal["block", "allow", "log_only"]

# Supported language tags for the blocked_languages multiselect dropdown.
# Only canonical names are listed; LANGUAGE_ALIASES in the guardrail normalizes
# aliases (e.g. js→javascript, sh→bash) when matching.
BLOCKED_LANGUAGES_OPTIONS = [
    "python",
    "javascript",
    "typescript",
    "bash",
    "ruby",
    "go",
    "java",
    "csharp",
    "php",
    "c",
    "cpp",
    "rust",
    "sql",
]


class CodeBlockDetection(TypedDict, total=False):
    """Detection output for a single fenced code block (for tracing/logging)."""

    type: Literal["code_block"]
    language: str
    confidence: float
    action_taken: CodeBlockActionTaken
    evidence: Optional[str]
    snippet: Optional[str]


class BlockCodeExecutionGuardrailConfigModel(GuardrailConfigModel):
    """Configuration for the Block Code Execution guardrail."""

    blocked_languages: Optional[List[str]] = Field(
        default=None,
        description="Language tags to block (e.g. python, javascript, bash). Empty or None = block all fenced code blocks.",
        json_schema_extra=cast(
            Any,
            {"ui_type": "multiselect", "options": BLOCKED_LANGUAGES_OPTIONS},
        ),
    )
    action: Literal["block", "mask"] = Field(
        default="block",
        description="'block' raises an error; 'mask' replaces the code block with a placeholder.",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Only block or mask when detection confidence >= this value; below threshold, allow or log_only.",
        json_schema_extra=cast(
            Any,
            {
                "ui_type": "percentage",
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "default_value": 0.5,
            },
        ),
    )
    detect_execution_intent: bool = Field(
        default=True,
        description="When True, block only when user intent is to run/execute; allow when intent is explain/refactor/don't run. Also block text-only execution requests (e.g. 'run `ls`', 'read /etc/passwd').",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Block Code Execution"
