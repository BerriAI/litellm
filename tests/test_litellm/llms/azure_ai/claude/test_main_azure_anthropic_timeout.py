"""
Ensure litellm.completion() forwards timeout to Azure Anthropic handler (main.py dispatch).
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm import completion
from litellm.types.utils import ModelResponse


def test_main_azure_ai_claude_completion_passes_timeout_to_azure_anthropic_handler():
    captured: dict = {}

    def fake_azure_anthropic_completion(**kwargs):
        captured.update(kwargs)
        return ModelResponse()

    with patch(
        "litellm.main.azure_anthropic_chat_completions"
    ) as mock_azure_anthropic:
        mock_azure_anthropic.completion = MagicMock(
            side_effect=fake_azure_anthropic_completion
        )

        completion(
            model="azure_ai/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://example.services.ai.azure.com/anthropic",
            api_key="test-key",
            timeout=42.5,
        )

    mock_azure_anthropic.completion.assert_called_once()
    assert captured["timeout"] == 42.5
    assert captured["model"] == "claude-sonnet-4-5"
    assert captured["custom_llm_provider"] == "azure_ai"
