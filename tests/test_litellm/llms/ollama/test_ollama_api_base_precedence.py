"""Regression tests for ollama `api_base` precedence.

Explicit `api_base` kwargs must override `litellm.api_base` global, across:
  - ollama / ollama_chat completion
  - ollama embedding (sync)
  - ollama aembedding (async)

Fixes the precedence bug in `litellm.main` where `litellm.api_base or api_base`
swallowed explicit kwargs whenever the global was set.
"""
from unittest.mock import MagicMock, patch

import pytest

import litellm


@pytest.mark.parametrize("model", ["ollama/phi", "ollama_chat/phi"])
def test_ollama_completion_explicit_api_base_takes_precedence(monkeypatch, model):
    monkeypatch.setattr(litellm, "api_base", "https://api.deepseek.com")

    with patch("litellm.main.base_llm_http_handler.completion") as mock_completion:
        mock_completion.return_value = MagicMock()

        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            api_base="http://localhost:11434",
        )

        assert mock_completion.call_args.kwargs["api_base"] == "http://localhost:11434"


def test_ollama_embedding_explicit_api_base_takes_precedence(monkeypatch):
    monkeypatch.setattr(litellm, "api_base", "https://api.deepseek.com")

    with patch("litellm.main.ollama.ollama_embeddings") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()

        litellm.embedding(
            model="ollama/qwen3-embedding:0.6b",
            input="hello",
            api_base="http://localhost:11434",
        )

        assert mock_embeddings.call_args.kwargs["api_base"] == "http://localhost:11434"


@pytest.mark.asyncio
async def test_ollama_aembedding_explicit_api_base_takes_precedence(monkeypatch):
    monkeypatch.setattr(litellm, "api_base", "https://api.deepseek.com")

    with patch("litellm.main.ollama.ollama_aembeddings") as mock_aembeddings:
        mock_aembeddings.return_value = MagicMock()

        await litellm.aembedding(
            model="ollama/qwen3-embedding:0.6b",
            input="hello",
            api_base="http://localhost:11434",
        )

        assert mock_aembeddings.call_args.kwargs["api_base"] == "http://localhost:11434"
