import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from base_rerank_unit_tests import BaseLLMRerankTest
import litellm


class TestInfinity(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.INFINITY

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "infinity/BAAI/bge-reranker-v2-m3",
        }
