"""Anthropic tool formatting used by LiteLLM's local token counter."""

import json
from collections.abc import Mapping


def format_anthropic_tool_for_token_counting(tool: Mapping[str, object]) -> str | None:
    """Keep a complete Anthropic tool schema in the text sent to the tokenizer."""
    if "input_schema" not in tool:
        return None
    return json.dumps(tool, ensure_ascii=False)
