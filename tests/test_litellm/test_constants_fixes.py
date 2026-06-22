"""Tests for constants.py fixes (issue #25140)."""

import ast
import re


def test_clarifai_models_no_implicit_concatenation():
    """Ensure clarifai_models entries are properly comma-separated (no implicit string concatenation)."""
    with open("litellm/constants.py") as f:
        content = f.read()

    # The two strings that were previously concatenated due to missing comma
    assert "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507" in content
    assert "clarifai/openai.chat-completion.gpt-5-nano" in content

    # Verify comma exists between them
    pattern = r'"clarifai/qwen\.qwenLM\.Qwen3-30B-A3B-Thinking-2507"\s*,\s*\n\s*"clarifai/openai\.chat-completion\.gpt-5-nano"'
    assert re.search(pattern, content), "Missing comma between clarifai model entries"


def test_azure_computer_use_cost_defaults():
    """Ensure Azure Computer Use cost defaults match documented values ($0.003/$0.012 per 1K tokens)."""
    with open("litellm/constants.py") as f:
        content = f.read()

    assert '"AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS", 0.003' in content
    assert '"AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS", 0.012' in content


def test_no_duplicate_max_size_per_item_in_memory_cache():
    """Ensure MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB is defined only once."""
    with open("litellm/constants.py") as f:
        content = f.read()

    definitions = re.findall(
        r"^MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB\s*=", content, re.MULTILINE
    )
    assert len(definitions) == 1, (
        f"Expected 1 definition, found {len(definitions)}"
    )
