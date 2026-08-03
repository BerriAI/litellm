"""
xAI Realtime API E2E Tests

Tests xAI's Grok Voice Agent API through LiteLLM's realtime interface.
Uses the base test class to ensure consistent behavior across providers.
"""

import os
import sys
from typing import Tuple

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from tests.llm_translation.realtime.base_realtime_tests import BaseRealtimeTest


class TestXAIRealtime(BaseRealtimeTest):
    """
    E2E tests for xAI Realtime API.

    xAI's Grok Voice Agent API is OpenAI-compatible:
    - Endpoint: wss://api.x.ai/v1/realtime
    - Model: grok-4-1-fast-non-reasoning
    - Initial event: historically "conversation.created"; xAI has since shipped
      "session.created" (matching OpenAI). Accept either to avoid spurious
      failures whenever xAI flips the wire format.
    """

    def get_model(self) -> str:
        return "xai/grok-4-1-fast-non-reasoning"

    def get_api_key_env_var(self) -> str:
        return "XAI_API_KEY"

    def get_initial_event_type(self) -> Tuple[str, ...]:
        return ("conversation.created", "session.created")
