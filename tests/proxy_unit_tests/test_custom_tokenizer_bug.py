"""
Regression tests for the proxy token_counter custom_tokenizer bug.

Bug: model_info was never populated from the matched deployment, so
custom_tokenizer was always None and token counting silently fell back to the
OpenAI tokenizer instead of the configured HuggingFace tokenizer.

The HuggingFace download boundary (Tokenizer.from_pretrained) is mocked so these
stay hermetic unit tests; the proxy's extraction-and-selection path runs for real.
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
import litellm.proxy.proxy_server
import litellm.utils
from litellm import Router
from litellm.proxy._types import TokenCountRequest
from litellm.proxy.proxy_server import token_counter


def _fake_hf_tokenizer(num_tokens: int) -> MagicMock:
    encoding = MagicMock()
    encoding.ids = list(range(num_tokens))
    tokenizer = MagicMock()
    tokenizer.encode.return_value = encoding
    return tokenizer


@pytest.mark.asyncio
async def test_custom_tokenizer_from_model_info_is_used(monkeypatch):
    """
    A deployment carrying model_info.custom_tokenizer must load and use that
    tokenizer. The model name deliberately matches no built-in HuggingFace
    tokenizer, so without the fix the response would fall back to
    "openai_tokenizer" and from_pretrained would never see the configured id.
    """
    llm_router = Router(
        model_list=[
            {
                "model_name": "my-embedding-model",
                "litellm_params": {
                    "model": "openai/self-hosted-embedder",
                    "api_base": "http://localhost:8080/v1",
                },
                "model_info": {
                    "mode": "embedding",
                    "custom_tokenizer": {
                        "identifier": "my-org/custom-tokenizer",
                        "revision": "v2",
                        "auth_token": None,
                    },
                },
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    with patch.object(litellm.utils, "Tokenizer") as mock_tokenizer_cls:
        mock_tokenizer_cls.from_pretrained.return_value = _fake_hf_tokenizer(7)

        response = await token_counter(
            request=TokenCountRequest(
                model="my-embedding-model",
                messages=[{"role": "user", "content": "Bonjour le monde"}],
            )
        )

    mock_tokenizer_cls.from_pretrained.assert_called_once_with(
        "my-org/custom-tokenizer", revision="v2", auth_token=None
    )
    assert response.tokenizer_type == "huggingface_tokenizer"
    assert response.request_model == "my-embedding-model"
    assert response.model_used == "self-hosted-embedder"
    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_model_without_custom_tokenizer_uses_default(monkeypatch):
    """
    Control: a deployment with no custom_tokenizer must not touch HuggingFace and
    must report the default OpenAI tokenizer.
    """
    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {},
            }
        ]
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    with patch.object(litellm.utils, "Tokenizer") as mock_tokenizer_cls:
        response = await token_counter(
            request=TokenCountRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )
        )

    mock_tokenizer_cls.from_pretrained.assert_not_called()
    assert response.tokenizer_type == "openai_tokenizer"
    assert response.model_used == "gpt-4"
    assert response.total_tokens > 0
