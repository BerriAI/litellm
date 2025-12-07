"""
Generic tool call parsing for Bedrock models with proprietary token formats.

Some Bedrock models (Kimi-K2, MiniMax M2) return tool calls as proprietary
tokens embedded in text rather than structured toolUse blocks. This module
parses and cleans those tokens.
"""

import re
from typing import Dict, List, Optional, Tuple

from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)


# =============================================================================
# Pattern Definitions
# =============================================================================

# Tool call marker patterns for models with proprietary formats
TOOL_CALL_PATTERNS: Dict[str, Dict[str, str]] = {
    # Kimi-K2-Thinking format
    "kimi": {
        "begin": "<|tool_call_begin|>",
        "arg_begin": "<|tool_call_argument_begin|>",
        "end": "<|tool_call_end|>",
        "section_begin": "<|tool_calls_section_begin|>",
        "section_end": "<|tool_calls_section_end|>",
    },
    # MiniMax M2 format (if discovered)
    "minimax": {
        "begin": "[TOOL_CALL]",
        "arg_begin": "=>",
        "end": "[/TOOL_CALL]",
    },
}

# Model ID -> Pattern name mapping
MODEL_TO_PATTERN: Dict[str, str] = {
    "moonshot.kimi-k2-thinking": "kimi",
    "minimax.minimax-m2": "minimax",
}


# =============================================================================
# Core Functions
# =============================================================================


def get_pattern_for_model(model: str) -> Optional[str]:
    """Get the pattern name for a model, or None if no parsing needed."""
    model_lower = model.lower()
    for model_id, pattern_name in MODEL_TO_PATTERN.items():
        if model_id in model_lower:
            return pattern_name
    return None


def clean_tool_tokens_from_text(text: str, model: str) -> str:
    """
    Remove proprietary tool call tokens from text content.

    This is the main function called from converse_transformation.py
    to clean model responses before returning to users.

    Args:
        text: The raw text content from model response
        model: The model identifier

    Returns:
        Cleaned text with proprietary tokens removed
    """
    pattern_name = get_pattern_for_model(model)
    if pattern_name is None:
        return text

    if pattern_name not in TOOL_CALL_PATTERNS:
        return text

    markers = TOOL_CALL_PATTERNS[pattern_name]
    cleaned = text

    # Remove section markers if present (Kimi)
    if "section_begin" in markers:
        section_pattern = (
            rf"{re.escape(markers['section_begin'])}"
            rf".*?"
            rf"{re.escape(markers['section_end'])}"
        )
        cleaned = re.sub(section_pattern, "", cleaned, flags=re.DOTALL)

    # Remove individual tool call blocks
    if pattern_name == "kimi":
        # Kimi format: <|tool_call_begin|>func<|tool_call_argument_begin|>{...}<|tool_call_end|>
        tool_pattern = (
            rf"{re.escape(markers['begin'])}"
            rf"[^<]*"
            rf"{re.escape(markers['arg_begin'])}"
            rf"[^<]*"
            rf"{re.escape(markers['end'])}"
        )
        cleaned = re.sub(tool_pattern, "", cleaned, flags=re.DOTALL)
    elif pattern_name == "minimax":
        # MiniMax format: [TOOL_CALL] {...} [/TOOL_CALL]
        tool_pattern = (
            rf"{re.escape(markers['begin'])}" rf".*?" rf"{re.escape(markers['end'])}"
        )
        cleaned = re.sub(tool_pattern, "", cleaned, flags=re.DOTALL)

    # Clean up extra whitespace
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned


def parse_tool_calls_for_model(
    model: str, text: str
) -> Tuple[str, List[ChatCompletionToolCallChunk]]:
    """
    Parse tool calls from text and return cleaned text.

    Args:
        model: Model identifier
        text: Raw text that may contain tool call markers

    Returns:
        Tuple of (cleaned_text, tool_calls)
    """
    pattern_name = get_pattern_for_model(model)
    if pattern_name is None:
        return text, []

    if pattern_name not in TOOL_CALL_PATTERNS:
        return text, []

    markers = TOOL_CALL_PATTERNS[pattern_name]
    tool_calls = []

    if pattern_name == "kimi":
        # Parse Kimi format
        pattern = (
            rf"{re.escape(markers['begin'])}"
            rf"\s*([\w\.:\-]+)\s*"
            rf"{re.escape(markers['arg_begin'])}"
            rf"\s*(\{{.*?\}})\s*"
            rf"{re.escape(markers['end'])}"
        )

        for idx, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
            func_name = match.group(1).strip()
            # Clean function name (remove prefixes like "functions.")
            if "." in func_name and func_name.startswith("functions."):
                func_name = func_name.replace("functions.", "").split(":")[0]

            args_json = match.group(2).strip()

            tool_calls.append(
                ChatCompletionToolCallChunk(
                    id=f"call_{idx}",
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=func_name, arguments=args_json
                    ),
                    index=idx,
                )
            )

    elif pattern_name == "minimax":
        # Parse MiniMax format: [TOOL_CALL] {tool => "name", args => {...}} [/TOOL_CALL]
        pattern = (
            rf"{re.escape(markers['begin'])}"
            rf"\s*\{{\s*tool\s*=>\s*\"([^\"]+)\"\s*,\s*args\s*=>\s*(\{{.*?\}})\s*\}}\s*"
            rf"{re.escape(markers['end'])}"
        )

        for idx, match in enumerate(re.finditer(pattern, text, re.DOTALL)):
            func_name = match.group(1).strip()
            args_json = match.group(2).strip()

            tool_calls.append(
                ChatCompletionToolCallChunk(
                    id=f"call_{idx}",
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=func_name, arguments=args_json
                    ),
                    index=idx,
                )
            )

    # Clean the text
    cleaned_text = clean_tool_tokens_from_text(text, model)

    return cleaned_text, tool_calls


# =============================================================================
# Model Detection
# =============================================================================


def is_kimi_model(model: str) -> bool:
    """Check if model is Kimi-K2-Thinking."""
    return "moonshot.kimi-k2-thinking" in model.lower()


def is_minimax_m2_model(model: str) -> bool:
    """Check if model is MiniMax M2."""
    return "minimax.minimax-m2" in model.lower()


def needs_token_cleaning(model: str) -> bool:
    """Check if model may return proprietary tokens that need cleaning."""
    return get_pattern_for_model(model) is not None
