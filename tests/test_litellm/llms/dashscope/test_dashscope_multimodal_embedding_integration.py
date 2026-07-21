"""
Integration tests for DashScope multimodal embedding end-to-end flow.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.dashscope.common_utils import DashScopeError
from litellm.llms.dashscope.embed.transformation_multimodal import DEFAULT_API_BASE
from litellm.types.utils import EmbeddingResponse

MOCK_SUCCESS_RESPONSE = {
    "output": {
        "embeddings": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3], "type": "text"}
        ]
    },
    "usage": {"input_tokens": 10, "total_tokens": 10},
    "request_id": "mock-request-id",
}

MOCK_ERROR_RESPONSE = {
    "code": "InvalidParameter",
    "message": "dimension must be between 1 and 2048",
    "request_id": "mock-request-id",
}


def _make_mock_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("POST", DEFAULT_API_BASE),
    )


# === 1. 同步完整流程 ===


def test_sync_multimodal_embedding():
    handler = BaseLLMHTTPHandler()
    mock_response = _make_mock_response(MOCK_SUCCESS_RESPONSE)

    mock_http_handler = MagicMock()
    mock_http_handler.post.return_value = mock_response

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_http_handler,
    ):
        with patch(
            "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
            return_value=None,
        ):
            result = handler.embedding(
                model="tongyi-embedding-vision-flash",
                input=["hello world"],
                timeout=30.0,
                custom_llm_provider="dashscope",
                logging_obj=MagicMock(),
                api_base=None,
                optional_params={},
                litellm_params={},
                model_response=EmbeddingResponse(),
                api_key="sk-test",
            )

    call_args = mock_http_handler.post.call_args
    assert "/multimodal-embedding" in call_args.kwargs.get("url", call_args.args[0] if call_args.args else "")
    assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert result.usage.prompt_tokens == 10
    assert result.id == "mock-request-id"


# === 2. 异步完整流程 ===


def test_async_multimodal_embedding():
    async def _run():
        handler = BaseLLMHTTPHandler()
        mock_response = _make_mock_response(MOCK_SUCCESS_RESPONSE)

        mock_async_handler = AsyncMock()
        mock_async_handler.post.return_value = mock_response

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
            return_value=mock_async_handler,
        ):
            with patch(
                "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
                return_value=None,
            ):
                result = await handler.aembedding(
                    request_data={
                        "model": "tongyi-embedding-vision-flash",
                        "input": {"contents": [{"text": "hello world"}]},
                    },
                    api_base=DEFAULT_API_BASE,
                    headers={
                        "Authorization": "Bearer sk-test",
                        "Content-Type": "application/json",
                    },
                    model="tongyi-embedding-vision-flash",
                    custom_llm_provider="dashscope",
                    provider_config=litellm.utils.ProviderConfigManager.get_provider_embedding_config(
                        model="tongyi-embedding-vision-flash",
                        provider=litellm.LlmProviders.DASHSCOPE,
                    ),
                    model_response=EmbeddingResponse(),
                    logging_obj=MagicMock(),
                    api_key="sk-test",
                    timeout=30.0,
                    optional_params={},
                    litellm_params={},
                )
        return result

    result = asyncio.run(_run())
    assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert result.usage.total_tokens == 10


# === 3. 错误响应 ===


def test_multimodal_embedding_api_error():
    handler = BaseLLMHTTPHandler()
    mock_response = _make_mock_response(MOCK_ERROR_RESPONSE, status_code=400)

    mock_http_handler = MagicMock()
    mock_http_handler.post.return_value = mock_response

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_http_handler,
    ):
        with patch(
            "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(DashScopeError) as exc:
                handler.embedding(
                    model="tongyi-embedding-vision-flash",
                    input=["hello world"],
                    timeout=30.0,
                    custom_llm_provider="dashscope",
                    logging_obj=MagicMock(),
                    api_base=None,
                    optional_params={},
                    litellm_params={},
                    model_response=EmbeddingResponse(),
                    api_key="sk-test",
                )
    assert exc.value.status_code == 400
    assert "dimension must be between 1 and 2048" in exc.value.message


# === 4. 自定义 api_base ===


def test_multimodal_embedding_custom_api_base():
    handler = BaseLLMHTTPHandler()
    mock_response = _make_mock_response(MOCK_SUCCESS_RESPONSE)

    mock_http_handler = MagicMock()
    mock_http_handler.post.return_value = mock_response

    vpc_base = "https://vpc.maas.aliyuncs.com/api/v1"

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_http_handler,
    ):
        with patch(
            "litellm.llms.dashscope.embed.transformation_multimodal.get_secret_str",
            return_value=None,
        ):
            result = handler.embedding(
                model="tongyi-embedding-vision-flash",
                input=["hello world"],
                timeout=30.0,
                custom_llm_provider="dashscope",
                logging_obj=MagicMock(),
                api_base=vpc_base,
                optional_params={},
                litellm_params={},
                model_response=EmbeddingResponse(),
                api_key="sk-test",
            )

    call_args = mock_http_handler.post.call_args
    posted_url = call_args.kwargs.get("url", call_args.args[0] if call_args.args else "")
    assert posted_url == f"{vpc_base}/services/embeddings/multimodal-embedding/multimodal-embedding"
    assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
