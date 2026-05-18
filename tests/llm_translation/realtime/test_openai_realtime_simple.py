"""
OpenAI Realtime API E2E Tests (using base class)

Tests OpenAI's Realtime API through LiteLLM's realtime interface.
Uses the base test class to ensure consistent behavior across providers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from tests.llm_translation.realtime.base_realtime_tests import BaseRealtimeTest


class TestOpenAIRealtime(BaseRealtimeTest):
    """
    E2E tests for OpenAI Realtime API using base test class.
    """

    def get_model(self) -> str:
        # gpt-4o-realtime-preview and all its date snapshots were removed by
        # OpenAI on 2026-05-18 ("model_not_found"). gpt-realtime is the
        # current GA alias.
        return "gpt-realtime"

    def get_api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"

    def get_initial_event_type(self) -> str:
        return "session.created"
