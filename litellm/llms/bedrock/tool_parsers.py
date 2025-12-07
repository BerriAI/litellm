"""
Generic tool call parsing system for Bedrock models.

This module provides a registry-based system for handling proprietary
tool call formats from different model providers.
"""

import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)

if TYPE_CHECKING:
    from litellm.types.llms.bedrock import BedrockConverseReasoningContentBlock

# Parser registry: model_pattern -> parser_function
PARSER_REGISTRY: Dict[
    str, Callable[[str, Any], tuple[str, List[ChatCompletionToolCallChunk]]]
] = {}


class ToolCallParser:
    """Base class for model-specific tool call parsers"""

    model_pattern: str = ""

    @classmethod
    def matches(cls, model: str) -> bool:
        """Check if this parser matches the given model"""
        return cls.model_pattern in model.lower()

    @classmethod
    def parse(
        cls, text: str, **kwargs
    ) -> tuple[str, List[ChatCompletionToolCallChunk]]:
        """Parse text and return (cleaned_text, tool_calls)"""
        raise NotImplementedError


class KimiK2ToolCallParser(ToolCallParser):
    """Parser for Kimi-K2-Thinking proprietary token format"""

    model_pattern = "moonshot.kimi-k2-thinking"

    @classmethod
    def parse(
        cls, text: str, **kwargs
    ) -> tuple[str, List[ChatCompletionToolCallChunk]]:
        """
        Parse Kimi-K2-Thinking proprietary token format.

        Markers: <|tool_call_begin|>, <|tool_call_argument_begin|>, <|tool_call_end|>
        Supports: <|tool_calls_section_begin|>...<|tool_calls_section_end|> wrapper
        """
        import json
        import re

        cleaned_text = ""
        parsed_tools = []

        # Regex to match Kimi tool call blocks
        # Pattern: <|tool_call_begin|>function_name<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
        tool_call_pattern = r"<\|tool_call_begin\|>([^<]+)<\|tool_call_argument_begin\|>([^<]+)<\|tool_call_end\|>"

        # Find all tool call matches
        matches = re.finditer(tool_call_pattern, text)

        last_end = 0
        tool_index = 0

        for match in matches:
            # Add text before this tool call to cleaned_text
            cleaned_text += text[last_end : match.start()]

            # Extract function name and arguments
            func_name = match.group(1).strip()
            args_json = match.group(2).strip()

            try:
                # Parse the arguments JSON (for validation)
                json.loads(args_json)

                # Create OpenAI-compatible tool call chunk
                function_chunk = ChatCompletionToolCallFunctionChunk(
                    name=func_name, arguments=args_json
                )

                tool_chunk = ChatCompletionToolCallChunk(
                    id=f"call_{tool_index}",
                    type="function",
                    function=function_chunk,
                    index=tool_index,
                )

                parsed_tools.append(tool_chunk)
                tool_index += 1

                # Add a placeholder in the cleaned text indicating where the tool was
                cleaned_text += f"[Tool call: {func_name}]"

            except json.JSONDecodeError as e:
                # If JSON parsing fails, just add the raw text
                cleaned_text += match.group(0)

            last_end = match.end()

        # Add any remaining text after the last tool call
        cleaned_text += text[last_end:]

        # If no tool calls were found, return the original text
        if not parsed_tools:
            return text, []

        return cleaned_text, parsed_tools


# Register built-in parsers
def register_parser(parser_class: type):
    """Register a tool call parser for a specific model"""
    PARSER_REGISTRY[parser_class.model_pattern] = parser_class.parse
    return parser_class


# Auto-register all built-in parsers
register_parser(KimiK2ToolCallParser)


def get_tool_call_parser(
    model: str,
) -> Optional[Callable[[str], tuple[str, List[ChatCompletionToolCallChunk]]]]:
    """
    Get an appropriate tool call parser for the given model.

    Args:
        model: The model identifier

    Returns:
        A parser function if found, None otherwise
    """
    for pattern, parser in PARSER_REGISTRY.items():
        if pattern in model.lower():
            return parser

    return None


def parse_tool_calls_for_model(
    model: str, text: str, **kwargs
) -> tuple[str, List[ChatCompletionToolCallChunk]]:
    """
    Parse tool calls using the appropriate model-specific parser.

    Args:
        model: The model identifier
        text: Text content that may contain proprietary tool call markers
        **kwargs: Additional arguments for the parser

    Returns:
        Tuple of (cleaned_text_without_markers, list_of_parsed_tools)
    """
    parser = get_tool_call_parser(model)

    if parser is None:
        # No specific parser found, return original text with no tools
        return text, []

    return parser(text, **kwargs)


def is_kimi_model(model: str) -> bool:
    """
    Check if the model is a Kimi-K2-Thinking model.

    Args:
        model: The model identifier

    Returns:
        True if it's a Kimi model, False otherwise
    """
    return "moonshot.kimi-k2-thinking" in model.lower()


def is_minimax_m2_model(model: str) -> bool:
    """
    Check if the model is a MiniMax M2 model.

    Args:
        model: The model identifier

    Returns:
        True if it's a MiniMax M2 model, False otherwise
    """
    return "minimax.minimax-m2" in model.lower()


def needs_kimi_parsing(text_content: str) -> bool:
    """
    Check if the text content contains Kimi-specific tool call markers.

    Args:
        text_content: The text content to check

    Returns:
        True if Kimi parsing is needed, False otherwise
    """
    return (
        "<|tool_calls_section_begin|>" in text_content
        or "<|tool_call_begin|>" in text_content
    )


def parse_kimi_tool_calls(text_content: str) -> List[ChatCompletionToolCallChunk]:
    """
    Parse Kimi-K2-Thinking proprietary token-stream format to extract tool calls.

    Kimi-K2-Thinking outputs raw tokens with markers like:
    <|tool_calls_section_begin|>...<|tool_call_begin|>function_name<|tool_call_argument_begin|>{...}<|tool_call_end|>...<|tool_calls_section_end|>

    Args:
        text_content: The text content from the model response

    Returns:
        List of parsed tool call chunks
    """
    tool_calls = []

    # Pattern to capture individual tool calls
    # Matches: <|tool_call_begin|>function_name<|tool_call_argument_begin|>{arguments}<|tool_call_end|>
    pattern = r"<\|tool_call_begin\|>\s*([\w\.]+)\s*<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>"

    matches = re.finditer(pattern, text_content, re.DOTALL)

    for i, match in enumerate(matches):
        function_name = match.group(1)
        arguments = match.group(2)

        # Generate a unique ID for the tool call
        tool_call_id = f"call_kimi_{i}_{hash(arguments) % 100000000}"

        # Create the function chunk
        function_chunk = ChatCompletionToolCallFunctionChunk(
            name=function_name, arguments=arguments
        )

        # Create the tool call chunk
        tool_call_chunk = ChatCompletionToolCallChunk(
            id=tool_call_id, type="function", function=function_chunk, index=i
        )

        tool_calls.append(tool_call_chunk)

    return tool_calls


# Export public API
__all__ = [
    "ToolCallParser",
    "KimiK2ToolCallParser",
    "register_parser",
    "get_tool_call_parser",
    "parse_tool_calls_for_model",
    "is_kimi_model",
    "is_minimax_m2_model",
    "needs_kimi_parsing",
    "parse_kimi_tool_calls",
]
