import os
import sys

import pytest

from litellm.llms.hosted_vllm.rerank.transformation import HostedVLLMRerankConfig
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class TestHostedVLLMRerankTransform:
    def setup_method(self):
        self.config = HostedVLLMRerankConfig()
        self.model = "hosted-vllm-model"

    def test_map_cohere_rerank_params_basic(self):
        params = self.config.map_cohere_rerank_params(
            non_default_params=None,
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
            top_n=2,
            rank_fields=["field1"],
            return_documents=True,
        )
        assert params["query"] == "test query"
        assert params["documents"] == ["doc1", "doc2"]
        assert params["top_n"] == 2
        assert params["rank_fields"] == ["field1"]
        assert params["return_documents"] is True

    def test_map_cohere_rerank_params_raises_on_max_chunks_per_doc(self):
        with pytest.raises(
            ValueError, match="Hosted VLLM does not support max_chunks_per_doc"
        ):
            self.config.map_cohere_rerank_params(
                non_default_params=None,
                model=self.model,
                drop_params=False,
                query="test query",
                documents=["doc1"],
                max_chunks_per_doc=5,
            )

    def test_get_complete_url(self):
        base = "https://api.example.com"
        url = self.config.get_complete_url(base, self.model)
        assert url == "https://api.example.com/rerank"
        # Already ends with /rerank
        url2 = self.config.get_complete_url(
            "https://api.example.com/rerank", self.model
        )
        assert url2 == "https://api.example.com/rerank"
        # Raises if api_base is None
        with pytest.raises(ValueError):
            self.config.get_complete_url(None, self.model)

    def test_transform_response(self):
        response_dict = {
            "id": "abc123",
            "results": [
                {"index": 0, "relevance_score": 0.9, "document": {"text": "doc1 text"}},
                {"index": 1, "relevance_score": 0.7, "document": {"text": "doc2 text"}},
            ],
            "usage": {"total_tokens": 42},
        }
        result = self.config._transform_response(response_dict)
        assert result.id == "abc123"
        assert len(result.results) == 2
        assert result.results[0]["index"] == 0
        assert result.results[0]["relevance_score"] == 0.9
        assert result.results[0]["document"]["text"] == "doc1 text"
        assert result.meta["billed_units"]["total_tokens"] == 42
        assert result.meta["tokens"]["input_tokens"] == 42

    def test_transform_response_missing_results(self):
        response_dict = {"id": "abc123", "usage": {"total_tokens": 10}}
        with pytest.raises(ValueError, match="No results found in the response="):
            self.config._transform_response(response_dict)

    def test_transform_response_missing_required_fields(self):
        response_dict = {
            "id": "abc123",
            "results": [{"relevance_score": 0.5}],
            "usage": {"total_tokens": 10},
        }
        with pytest.raises(ValueError, match="Missing required fields in the result="):
            self.config._transform_response(response_dict)
