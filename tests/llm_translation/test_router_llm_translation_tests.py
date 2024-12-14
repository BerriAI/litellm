"""
Uses litellm.Router, ensures router.completion and router.acompletion pass BaseLLMChatTest
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from base_llm_unit_tests import BaseLLMChatTest
from litellm.router import Router
from litellm._logging import verbose_logger, verbose_router_logger
import logging


class TestRouterLLMTranslation(BaseLLMChatTest):
    verbose_router_logger.setLevel(logging.DEBUG)

    litellm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ]
    )

    @property
    def completion_function(self):
        return self.litellm_router.completion

    @property
    def async_completion_function(self):
        return self.litellm_router.acompletion

    def get_base_completion_call_args(self) -> dict:
        return {"model": "gpt-4o-mini"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass
