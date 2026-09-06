"""
Test transformation logic for Docker Model Runner rerank.

This test verifies that the DockerModelRunnerRerankConfig correctly transforms
rerank requests and handles both llama.cpp and vLLM response formats.
"""

import json
from unittest.mock import MagicMock

import pytest

from litellm.llms.docker_model_runner.rerank.transformation import (
    DockerModelRunnerRerankConfig,
)
from litellm.types.rerank import RerankResponse


class TestDockerModelRunnerRerankUrlGeneration:
    """Tests for rerank URL generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerRerankConfig()
        self.model = "ai/qwen3-reranker:0.6B"

    def test_get_complete_url_default_api_base(self):
        """Test URL generation with default api_base (no /engines prefix)."""
        url = self.config.get_complete_url(
            api_base=None,
            model=self.model,
            optional_params=None,
        )
        assert url == "http://localhost:12434/rerank"

    def test_get_complete_url_with_host_only(self):
        """Test URL generation with host-only api_base."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434",
            model=self.model,
            optional_params=None,
        )
        assert url == "http://localhost:12434/rerank"

    def test_get_complete_url_already_has_rerank_path(self):
        """Test URL generation when api_base already ends with /rerank."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/rerank",
            model=self.model,
            optional_params=None,
        )
        assert url == "http://localhost:12434/rerank"

    def test_get_complete_url_container_host(self):
        """Test URL generation from within a Docker container."""
        url = self.config.get_complete_url(
            api_base="http://model-runner.docker.internal",
            model=self.model,
            optional_params=None,
        )
        assert url == "http://model-runner.docker.internal/rerank"

    def test_get_complete_url_strips_v1_suffix(self):
        """api_base with the OpenAI-style /v1 suffix must not end up in the rerank URL."""
        url = self.config.get_complete_url(
            api_base="http://localhost:12434/engines/v1",
            model=self.model,
            optional_params=None,
        )
        assert url == "http://localhost:12434/engines/rerank"


class TestDockerModelRunnerRerankRequestTransformation:
    """Tests for rerank request transformation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerRerankConfig()
        self.model = "ai/qwen3-reranker:0.6B"

    def test_transform_rerank_request_basic(self):
        """Test basic rerank request transformation."""
        request = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params={
                "query": "What is the capital of France?",
                "documents": [
                    "The capital of Brazil is Brasilia.",
                    "The capital of France is Paris.",
                ],
                "top_n": 2,
            },
            headers={},
        )
        assert request["model"] == self.model
        assert request["query"] == "What is the capital of France?"
        assert len(request["documents"]) == 2
        assert request["top_n"] == 2

    def test_transform_rerank_request_missing_query_raises(self):
        """Test that missing query raises ValueError."""
        with pytest.raises(ValueError, match="query is required"):
            self.config.transform_rerank_request(
                model=self.model,
                optional_rerank_params={
                    "documents": ["doc1", "doc2"],
                },
                headers={},
            )

    def test_transform_rerank_request_missing_documents_raises(self):
        """Test that missing documents raises ValueError."""
        with pytest.raises(ValueError, match="documents is required"):
            self.config.transform_rerank_request(
                model=self.model,
                optional_rerank_params={
                    "query": "test query",
                },
                headers={},
            )

    def test_transform_rerank_request_with_top_n(self):
        """Test rerank request with top_n parameter."""
        request = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params={
                "query": "test query",
                "documents": ["doc1", "doc2", "doc3"],
                "top_n": 1,
            },
            headers={},
        )
        assert request["top_n"] == 1

    def test_transform_rerank_request_with_documents_only(self):
        """Test rerank request with only documents (no extra params)."""
        request = self.config.transform_rerank_request(
            model=self.model,
            optional_rerank_params={
                "query": "test query",
                "documents": ["doc1", "doc2"],
            },
            headers={},
        )
        assert "top_n" not in request
        assert "rank_fields" not in request
        assert "return_documents" not in request


class TestDockerModelRunnerRerankParamMapping:
    """Tests for mapping Cohere-style rerank params."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerRerankConfig()
        self.model = "ai/qwen3-reranker:0.6B"

    def test_map_cohere_rerank_params_basic(self):
        """Test that query/documents/top_n are mapped through."""
        mapped = self.config.map_cohere_rerank_params(
            non_default_params={},
            model=self.model,
            drop_params=False,
            query="test query",
            documents=["doc1", "doc2"],
            top_n=1,
        )
        assert mapped["query"] == "test query"
        assert mapped["documents"] == ["doc1", "doc2"]
        assert mapped["top_n"] == 1

    def test_map_cohere_rerank_params_rejects_max_chunks_per_doc(self):
        """Test that max_chunks_per_doc raises ValueError."""
        with pytest.raises(ValueError, match="max_chunks_per_doc"):
            self.config.map_cohere_rerank_params(
                non_default_params={},
                model=self.model,
                drop_params=False,
                query="test query",
                documents=["doc1"],
                max_chunks_per_doc=3,
            )


class TestDockerModelRunnerRerankResponseTransformation:
    """Tests for rerank response transformation (both llama.cpp and vLLM formats)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DockerModelRunnerRerankConfig()

    def test_transform_response_llamacpp_format(self):
        """
        Test rerank response transformation for llama.cpp format.

        llama.cpp returns:
        {
          "model": "ai/qwen3-reranker:0.6B",
          "object": "list",
          "usage": {"prompt_tokens": 257, "total_tokens": 257},
          "results": [{"index": 1, "relevance_score": 0.999}, ...]
        }
        """
        response_data = {
            "model": "ai/qwen3-reranker:0.6B",
            "object": "list",
            "usage": {"prompt_tokens": 257, "total_tokens": 257},
            "results": [
                {"index": 1, "relevance_score": 0.9997124671936035},
                {"index": 0, "relevance_score": 8.676401921547949e-05},
                {"index": 2, "relevance_score": 1.1414089385652915e-05},
            ],
        }

        result = self.config._transform_response(response_data)

        assert isinstance(result, RerankResponse)
        assert len(result.results) == 3
        assert result.results[0]["index"] == 1
        assert result.results[0]["relevance_score"] == pytest.approx(0.9997, abs=0.001)
        assert result.results[1]["index"] == 0
        assert result.meta["tokens"]["input_tokens"] == 257

    def test_transform_response_vllm_format(self):
        """
        Test rerank response transformation for vLLM format.

        vLLM returns:
        {
          "id": "rerank-...",
          "model": "sha256:...",
          "usage": {"total_tokens": 44},
          "results": [{"index": 1, "document": {"text": "..."}, "relevance_score": 0.98}, ...]
        }
        """
        response_data = {
            "id": "rerank-001dee508c0b490a8b6ae78fdb22bd60",
            "model": "sha256:9749a463b5ad73fad92f25bdaa4d5bf36de45edb780020f62de8eee4e6241f9f",
            "usage": {"total_tokens": 44},
            "results": [
                {
                    "index": 1,
                    "document": {"text": "The capital of France is Paris."},
                    "relevance_score": 0.9818331003189087,
                },
                {
                    "index": 0,
                    "document": {"text": "The capital of Brazil is Brasilia."},
                    "relevance_score": 0.8894031047821045,
                },
                {
                    "index": 2,
                    "document": {"text": "Horses and cows are both animals."},
                    "relevance_score": 0.7491137385368347,
                },
            ],
        }

        result = self.config._transform_response(response_data)

        assert isinstance(result, RerankResponse)
        assert result.id == "rerank-001dee508c0b490a8b6ae78fdb22bd60"
        assert len(result.results) == 3
        assert result.results[0]["index"] == 1
        assert result.results[0]["relevance_score"] == pytest.approx(0.9818, abs=0.001)
        assert result.results[0]["document"] is not None
        assert result.results[0]["document"]["text"] == "The capital of France is Paris."
        assert result.results[1]["document"]["text"] == "The capital of Brazil is Brasilia."
        assert result.results[2]["document"]["text"] == "Horses and cows are both animals."

    def test_transform_response_missing_results_raises(self):
        """Test that missing results raises ValueError."""
        response_data = {
            "model": "ai/qwen3-reranker:0.6B",
            "usage": {"total_tokens": 44},
        }

        with pytest.raises(ValueError, match="No results found in the response"):
            self.config._transform_response(response_data)

    def test_transform_response_missing_score_raises(self):
        """Test that missing relevance_score raises ValueError."""
        response_data = {
            "results": [{"index": 0}],
        }

        with pytest.raises(ValueError, match="Missing required fields in the result"):
            self.config._transform_response(response_data)

    def test_transform_rerank_response_via_http(self):
        """Test full rerank response transformation via HTTP mock."""
        response_data = {
            "id": "rerank-http-test",
            "model": "ai/qwen3-reranker:0.6B",
            "usage": {"total_tokens": 100},
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.05},
            ],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = response_data
        mock_response.text = json.dumps(response_data)

        model_response = MagicMock()

        result = self.config.transform_rerank_response(
            model="ai/qwen3-reranker:0.6B",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            api_key=None,
            request_data={},
            optional_params={},
            litellm_params={},
        )

        assert len(result.results) == 2
        assert result.results[0]["relevance_score"] == pytest.approx(0.95, abs=0.01)
        assert result.results[1]["relevance_score"] == pytest.approx(0.05, abs=0.01)

    def test_validate_environment_returns_headers(self):
        """Test that validate_environment returns proper headers."""
        headers = self.config.validate_environment(
            headers={},
            model="ai/qwen3-reranker:0.6B",
            api_key="test-key",
        )
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
