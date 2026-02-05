"""
xAI Realtime API E2E Tests

Tests xAI's Grok Voice Agent API through LiteLLM's realtime interface.
Uses the base test class to ensure consistent behavior across providers.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from tests.llm_translation.realtime.base_realtime_tests import BaseRealtimeTest


class TestXAIRealtime(BaseRealtimeTest):
    """
    E2E tests for xAI Realtime API.
    
    xAI's Grok Voice Agent API is OpenAI-compatible but uses:
    - Different initial event: "conversation.created" instead of "session.created"
    - Different endpoint: wss://api.x.ai/v1/realtime
    - Model: grok-4-1-fast-non-reasoning
    """
    
    def get_model(self) -> str:
        return "xai/grok-4-1-fast-non-reasoning"
    
    def get_api_key_env_var(self) -> str:
        return "XAI_API_KEY"
    
    def get_initial_event_type(self) -> str:
        return "conversation.created"
