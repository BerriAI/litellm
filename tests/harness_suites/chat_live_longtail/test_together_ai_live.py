"""
Test TogetherAI LLM
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from base_llm_unit_tests import BaseLLMChatTest


class TestTogetherAI(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm.set_verbose = True
        return {"model": "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"}
