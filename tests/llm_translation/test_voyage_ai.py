import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from base_embedding_unit_tests import BaseLLMEmbeddingTest
import litellm


class TestVoyageAI(BaseLLMEmbeddingTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.VOYAGE

    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "voyage/voyage-3-lite",
        }


def test_voyage_ai_embedding_extra_params():
    litellm.set_verbose = True
    response = litellm.embedding(
        model="voyage/voyage-3-lite",
        input=["a"],
    )
    print(response)
