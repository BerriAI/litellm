import os
import sys
from typing import cast

import httpx

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.siliconflow.rerank.transformation import SiliconFlowRerankConfig
from litellm.types.rerank import RerankResponse
from litellm.types.utils import ModelInfo


class LoggingStub:
    def post_call(self, original_response: str) -> None:
        return None


class TestSiliconFlowRerankConfig:
    def setup_method(self):
        self.config = SiliconFlowRerankConfig()

    def test_map_cohere_rerank_params(self):
        params = self.config.map_cohere_rerank_params(
            non_default_params={"top_n": 2, "return_documents": False},
            model="Pro/BAAI/bge-reranker-v2-m3",
            drop_params=False,
            query="hello",
            documents=["doc-1", "doc-2"],
            top_n=3,
            return_documents=True,
        )

        assert params["query"] == "hello"
        assert params["documents"] == ["doc-1", "doc-2"]
        assert params["top_n"] == 2
        assert params["return_documents"] is False

    def test_transform_rerank_response_and_cost(self):
        raw_response = httpx.Response(
            200,
            json={
                "id": "rerank-1",
                "results": [
                    {"index": 0, "relevance_score": 0.92, "document": {"text": "doc-1"}},
                    {"index": 1, "relevance_score": 0.41, "document": "doc-2"},
                ],
                "meta": {
                    "tokens": {"input_tokens": 64, "output_tokens": 0},
                    "billed_units": {"search_units": 1},
                },
            },
        )

        result = self.config.transform_rerank_response(
            model="Pro/BAAI/bge-reranker-v2-m3",
            raw_response=raw_response,
            model_response=RerankResponse(),
            logging_obj=cast(LiteLLMLoggingObj, LoggingStub()),
        )

        assert result.results is not None
        assert result.meta is not None
        assert "billed_units" in result.meta
        billed_units = result.meta["billed_units"]
        assert billed_units is not None
        first_document = cast(dict[str, str], result.results[0].get("document"))
        second_document = cast(dict[str, str], result.results[1].get("document"))
        assert result.id == "rerank-1"
        assert first_document["text"] == "doc-1"
        assert second_document["text"] == "doc-2"
        assert billed_units.get("total_tokens") == 64

        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model="Pro/BAAI/bge-reranker-v2-m3",
            billed_units=billed_units,
            model_info=cast(ModelInfo, {"input_cost_per_token": 9.8e-09}),
        )
        assert prompt_cost == 64 * 9.8e-09
        assert completion_cost == 0.0
