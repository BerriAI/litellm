"""
Unit tests for DashScope rerank transformation.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.dashscope.common_utils import DashScopeError
from litellm.llms.dashscope.rerank.transformation import (
    DEFAULT_RERANK_URL,
    DashScopeRerankConfig,
)
from litellm.types.rerank import RerankResponse


class TestDashScopeRerankURL:
    def setup_method(self):
        self.config = DashScopeRerankConfig()

    def test_default_url(self):
        url = self.config.get_complete_url(api_base=None, model="qwen3-rerank")
        assert url == DEFAULT_RERANK_URL

    def test_explicit_v1_base_appends_reranks(self):
        url = self.config.get_complete_url(
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3-rerank",
        )
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/reranks"

    def test_intl_v1_base_appends_reranks(self):
        url = self.config.get_complete_url(
            api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            model="qwen3-rerank",
        )
        assert url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/reranks"

    def test_already_complete_url_passthrough(self):
        full = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        assert self.config.get_complete_url(api_base=full, model="qwen3-rerank") == full

    def test_trailing_slash_stripped(self):
        full = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks/"
        assert self.config.get_complete_url(
            api_base=full, model="qwen3-rerank"
        ) == full.rstrip("/")

    def test_custom_v1_base_appends_reranks(self):
        url = self.config.get_complete_url(
            api_base="https://my-proxy.example.com/v1", model="qwen3-rerank"
        )
        assert url == "https://my-proxy.example.com/v1/reranks"


class TestDashScopeRerankRequest:
    def setup_method(self):
        self.config = DashScopeRerankConfig()

    def test_validate_environment_with_explicit_key(self):
        headers = self.config.validate_environment(
            headers={}, model="qwen3-rerank", api_key="sk-test"
        )
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"

    def test_validate_environment_missing_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            self.config.validate_environment(
                headers={}, model="qwen3-rerank", api_key=None
            )

    def test_validate_environment_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
        headers = self.config.validate_environment(
            headers={}, model="qwen3-rerank", api_key=None
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_supported_params(self):
        assert self.config.get_supported_cohere_rerank_params("qwen3-rerank") == [
            "query",
            "documents",
            "top_n",
            "return_documents",
        ]

    def test_map_params_drops_unsupported(self):
        # qwen3-rerank accepts query/documents/top_n/return_documents.
        # rank_fields and max_*_per_doc are silently dropped.
        params = self.config.map_cohere_rerank_params(
            non_default_params={},
            model="qwen3-rerank",
            drop_params=False,
            query="什么是文本排序模型",
            documents=["d1", "d2"],
            top_n=2,
            rank_fields=["title"],
            return_documents=True,
            max_chunks_per_doc=5,
            max_tokens_per_doc=100,
        )
        assert params == {
            "query": "什么是文本排序模型",
            "documents": ["d1", "d2"],
            "top_n": 2,
            "return_documents": True,
        }

    def test_transform_request_full(self):
        body = self.config.transform_rerank_request(
            model="qwen3-rerank",
            optional_rerank_params={
                "query": "如何制作美味的苹果派?",
                "documents": ["a", "b"],
                "top_n": 5,
                "return_documents": True,
            },
            headers={},
        )
        assert body == {
            "model": "qwen3-rerank",
            "query": "如何制作美味的苹果派?",
            "documents": ["a", "b"],
            "top_n": 5,
            "return_documents": True,
        }

    def test_transform_request_omits_unset_optional(self):
        body = self.config.transform_rerank_request(
            model="qwen3-rerank",
            optional_rerank_params={"query": "q", "documents": ["a"]},
            headers={},
        )
        assert "top_n" not in body
        assert "return_documents" not in body

    def test_transform_request_requires_query(self):
        with pytest.raises(ValueError, match="query"):
            self.config.transform_rerank_request(
                model="qwen3-rerank",
                optional_rerank_params={"documents": ["a"]},
                headers={},
            )

    def test_transform_request_requires_documents(self):
        with pytest.raises(ValueError, match="documents"):
            self.config.transform_rerank_request(
                model="qwen3-rerank",
                optional_rerank_params={"query": "q"},
                headers={},
            )


class TestDashScopeRerankResponse:
    def setup_method(self):
        self.config = DashScopeRerankConfig()
        self.logging = MagicMock()

    def _resp(self, body, status_code=200):
        return httpx.Response(
            status_code=status_code,
            content=json.dumps(body).encode(),
            request=httpx.Request("POST", "https://example.com"),
        )

    def test_success_response(self):
        body = {
            "object": "list",
            "results": [
                {"index": 0, "relevance_score": 0.93},
                {"index": 2, "relevance_score": 0.34},
            ],
            "model": "qwen3-rerank",
            "id": "85ba5752",
            "usage": {"total_tokens": 79},
        }
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
            api_key="sk",
            request_data={"query": "q"},
        )
        assert out.id == "85ba5752"
        assert out.results == [
            {"index": 0, "relevance_score": 0.93},
            {"index": 2, "relevance_score": 0.34},
        ]
        assert out.meta == {
            "billed_units": {"total_tokens": 79},
            "tokens": {"input_tokens": 79},
        }

    def test_response_with_return_documents_real_payload(self):
        # Verbatim sample from a real qwen3-rerank call with return_documents=true.
        body = {
            "object": "list",
            "results": [
                {
                    "document": {
                        "text": "苹果派的制作步骤包括准备面团、切苹果、调制馅料、组装和烘烤。"
                    },
                    "index": 1,
                    "relevance_score": 0.8304247466067356,
                },
                {
                    "document": {
                        "text": "制作苹果派时，预先煮软苹果可以缩短烘烤时间。"
                    },
                    "index": 3,
                    "relevance_score": 0.7142660211908354,
                },
            ],
            "model": "qwen3-rerank",
            "id": "e191b077-97c4-9929-b121-c2fbd2c7b0af",
            "usage": {"total_tokens": 192},
        }
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
            request_data={"query": "如何制作美味的苹果派?"},
        )
        assert out.id == "e191b077-97c4-9929-b121-c2fbd2c7b0af"
        assert out.results == [
            {
                "index": 1,
                "relevance_score": 0.8304247466067356,
                "document": {
                    "text": "苹果派的制作步骤包括准备面团、切苹果、调制馅料、组装和烘烤。"
                },
            },
            {
                "index": 3,
                "relevance_score": 0.7142660211908354,
                "document": {"text": "制作苹果派时，预先煮软苹果可以缩短烘烤时间。"},
            },
        ]
        assert out.meta == {
            "billed_units": {"total_tokens": 192},
            "tokens": {"input_tokens": 192},
        }

    def test_response_string_document_normalized(self):
        # Defensive path: if a future API revision returns a bare string,
        # normalize to {"text": ...} so downstream code stays consistent.
        body = {
            "results": [{"index": 0, "relevance_score": 0.9, "document": "hello"}],
            "model": "qwen3-rerank",
            "usage": {"total_tokens": 5},
        }
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
        )
        assert out.results[0]["document"] == {"text": "hello"}

    def test_missing_id_generates_uuid(self):
        body = {"results": [{"index": 0, "relevance_score": 0.5}], "usage": {}}
        out = self.config.transform_rerank_response(
            model="qwen3-rerank",
            raw_response=self._resp(body),
            model_response=RerankResponse(),
            logging_obj=self.logging,
        )
        assert out.id is not None and len(out.id) > 0

    def test_error_envelope_raises(self):
        body = {
            "code": "InvalidApiKey",
            "message": "Invalid API-key provided.",
            "request_id": "fb53",
        }
        with pytest.raises(DashScopeError) as exc_info:
            self.config.transform_rerank_response(
                model="qwen3-rerank",
                raw_response=self._resp(body, status_code=401),
                model_response=RerankResponse(),
                logging_obj=self.logging,
            )
        assert "Invalid API-key provided." in str(exc_info.value)

    def test_non_json_response_raises(self):
        bad = httpx.Response(
            status_code=500,
            content=b"<html>bad gateway</html>",
            request=httpx.Request("POST", "https://example.com"),
        )
        with pytest.raises(DashScopeError):
            self.config.transform_rerank_response(
                model="qwen3-rerank",
                raw_response=bad,
                model_response=RerankResponse(),
                logging_obj=self.logging,
            )

    def test_get_error_class(self):
        err = self.config.get_error_class(
            error_message="boom", status_code=500, headers={}
        )
        assert isinstance(err, DashScopeError)
        assert err.status_code == 500


class TestDashScopeRerankCost:
    def setup_method(self):
        self.config = DashScopeRerankConfig()

    def test_cost_with_total_tokens(self):
        from litellm.types.rerank import RerankBilledUnits

        model_info = {"input_cost_per_token": 1e-6, "output_cost_per_token": 0.0}
        billed = RerankBilledUnits(total_tokens=100)
        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model="qwen3-rerank",
            billed_units=billed,
            model_info=model_info,
        )
        assert prompt_cost == pytest.approx(1e-4)
        assert completion_cost == 0.0

    def test_cost_zero_tokens(self):
        from litellm.types.rerank import RerankBilledUnits

        model_info = {"input_cost_per_token": 1e-6, "output_cost_per_token": 0.0}
        billed = RerankBilledUnits(total_tokens=0)
        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model="qwen3-rerank",
            billed_units=billed,
            model_info=model_info,
        )
        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_cost_no_model_info(self):
        from litellm.types.rerank import RerankBilledUnits

        billed = RerankBilledUnits(total_tokens=100)
        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model="qwen3-rerank",
            billed_units=billed,
            model_info=None,
        )
        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_cost_no_billed_units(self):
        model_info = {"input_cost_per_token": 1e-6, "output_cost_per_token": 0.0}
        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model="qwen3-rerank",
            billed_units=None,
            model_info=model_info,
        )
        assert prompt_cost == 0.0
        assert completion_cost == 0.0

    def test_cost_missing_input_cost_per_token(self):
        from litellm.types.rerank import RerankBilledUnits

        model_info = {"output_cost_per_token": 0.0}
        billed = RerankBilledUnits(total_tokens=100)
        prompt_cost, completion_cost = self.config.calculate_rerank_cost(
            model="qwen3-rerank",
            billed_units=billed,
            model_info=model_info,
        )
        assert prompt_cost == 0.0
        assert completion_cost == 0.0


class TestProviderConfigManagerDispatch:
    def test_dashscope_returns_rerank_config(self):
        import litellm
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_rerank_config(
            model="qwen3-rerank",
            provider=litellm.LlmProviders.DASHSCOPE,
            api_base=None,
            present_version_params=[],
        )
        assert isinstance(cfg, DashScopeRerankConfig)
