import os
import sys
from typing import cast

import httpx

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.siliconflow.embed.transformation import SiliconFlowEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


class LoggingStub:
    def post_call(self, original_response: str) -> None:
        return None


class TestSiliconFlowEmbeddingConfig:
    def setup_method(self):
        self.config = SiliconFlowEmbeddingConfig()

    def test_transform_embedding_request(self):
        request_body = self.config.transform_embedding_request(
            model="BAAI/bge-m3",
            input=["hello"],
            optional_params={"encoding_format": "float"},
            headers={},
        )

        assert request_body == {
            "model": "BAAI/bge-m3",
            "input": ["hello"],
            "encoding_format": "float",
        }

    def test_transform_embedding_response_uses_input_tokens_when_prompt_tokens_missing(self):
        raw_response = httpx.Response(
            200,
            json={
                "object": "list",
                "model": "BAAI/bge-m3",
                "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
                "usage": {"input_tokens": 12},
            },
        )
        result = self.config.transform_embedding_response(
            model="BAAI/bge-m3",
            raw_response=raw_response,
            model_response=EmbeddingResponse(),
            logging_obj=cast(LiteLLMLoggingObj, LoggingStub()),
            api_key=None,
            request_data={},
            optional_params={},
            litellm_params={},
        )

        assert result.usage is not None
        assert result.object == "list"
        assert result.model == "BAAI/bge-m3"
        assert result.usage.prompt_tokens == 12
        assert result.usage.total_tokens == 12
