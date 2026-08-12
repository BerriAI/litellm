"""
Tests for Parallel AI chat transformation (OpenAI-compatible /chat/completions).

Source: litellm/llms/parallel_ai/chat/transformation.py
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import ModelResponse
from litellm.llms.parallel_ai.chat.transformation import (
    ParallelAIChatConfig,
    _citation_urls_from_basis,
)

MOCK_BASIS = [
    {
        "field": "answer",
        "citations": [
            {"url": "https://example.com/a", "excerpts": ["excerpt one"]},
            {"url": "https://example.com/b"},
        ],
        "reasoning": "compared both sources",
        "confidence": "high",
    },
    {
        "field": "answer.details",
        "citations": [{"url": "https://example.com/a"}],
        "reasoning": "same source",
        "confidence": "medium",
    },
]


class TestParallelAIChatConfig:
    def test_provider_info_defaults(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")

        api_base, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.parallel.ai"
        assert api_key == "pk-test"

    def test_provider_info_prefers_parallel_ai_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-primary")
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-fallback")

        _, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(None, None)
        assert api_key == "pk-primary"

    def test_provider_info_explicit_args_win(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")

        api_base, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(
            "https://proxy.example.com", "pk-explicit"
        )
        assert api_base == "https://proxy.example.com"
        assert api_key == "pk-explicit"

    def test_supported_params_exclude_sampling_and_tools(self):
        supported = ParallelAIChatConfig().get_supported_openai_params(model="core")
        assert "response_format" in supported
        assert "stream" in supported
        assert "temperature" not in supported
        assert "tools" not in supported
        assert "tool_choice" not in supported

    def test_get_llm_provider_routes_parallel_ai(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
        model, provider, api_key, api_base = litellm.get_llm_provider("parallel_ai/speed")
        assert model == "speed"
        assert provider == "parallel_ai"
        assert api_key == "pk-test"
        assert api_base == "https://api.parallel.ai"

    def test_transform_response_extracts_basis(self):
        config = ParallelAIChatConfig()
        raw_json = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "core",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "grounded answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "basis": MOCK_BASIS,
        }
        raw_response = MagicMock()
        raw_response.json.return_value = raw_json
        raw_response.headers = {}

        result = config.transform_response(
            model="core",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "question"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "grounded answer"
        assert getattr(result, "basis") == MOCK_BASIS
        assert getattr(result, "citations") == [
            "https://example.com/a",
            "https://example.com/b",
        ]

    def test_transform_response_without_basis(self):
        config = ParallelAIChatConfig()
        raw_json = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "speed",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "fast answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        raw_response = MagicMock()
        raw_response.json.return_value = raw_json
        raw_response.headers = {}

        result = config.transform_response(
            model="speed",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[{"role": "user", "content": "question"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "fast answer"
        assert not hasattr(result, "basis")
        assert not hasattr(result, "citations")


class TestCitationUrlsFromBasis:
    def test_dedupes_and_preserves_order(self):
        assert _citation_urls_from_basis(MOCK_BASIS) == [
            "https://example.com/a",
            "https://example.com/b",
        ]

    @pytest.mark.parametrize("bad_basis", [None, "not-a-list", {}, 42])
    def test_non_list_basis_returns_empty(self, bad_basis):
        assert _citation_urls_from_basis(bad_basis) == []

    def test_malformed_entries_are_skipped(self):
        basis = [
            "not-a-dict",
            {"citations": "not-a-list"},
            {"citations": [{"no_url": True}, {"url": ""}, {"url": "https://ok.example.com"}]},
        ]
        assert _citation_urls_from_basis(basis) == ["https://ok.example.com"]
