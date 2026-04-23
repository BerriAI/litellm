"""
Tests for enterprise/litellm_enterprise/proxy/hooks/managed_files.py

Regression test for afile_retrieve called without credentials in
async_post_call_success_hook when processing completed batch responses.
"""

import pytest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.llms.openai import OpenAIFileObject
from litellm.types.utils import LiteLLMBatch


def _make_file_object(file_id: str = "file-output-abc") -> OpenAIFileObject:
    return OpenAIFileObject(
        id=file_id,
        bytes=100,
        created_at=1700000000,
        filename="output.jsonl",
        object="file",
        purpose="batch_output",
        status="processed",
    )


def _make_batch_response(
    batch_id: str = "batch-123",
    output_file_id: Optional[str] = "file-output-abc",
    status: str = "completed",
    model_id: str = "model-deploy-xyz",
    model_name: str = "azure/gpt-4",
) -> LiteLLMBatch:
    """Create a LiteLLMBatch response with hidden params set as the router would."""
    batch = LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id="file-input-abc",
        object="batch",
        status=status,
        output_file_id=output_file_id,
    )
    batch._hidden_params = {
        "unified_file_id": "some-unified-id",
        "unified_batch_id": "some-unified-batch-id",
        "model_id": model_id,
        "model_name": model_name,
    }
    return batch


def _make_user_api_key_dict() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="test-user",
        parent_otel_span=None,
    )


def _make_managed_files_instance():
    """Create a _PROXY_LiteLLMManagedFiles with storage methods mocked out."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    mock_cache = MagicMock()
    mock_prisma = MagicMock()

    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=mock_cache,
        prisma_client=mock_prisma,
    )
    instance.store_unified_file_id = AsyncMock()
    instance.store_unified_object_id = AsyncMock()
    return instance


@pytest.mark.asyncio
async def test_should_pass_credentials_to_afile_retrieve():
    """
    When async_post_call_success_hook processes a completed batch with an output_file_id,
    it calls afile_retrieve to fetch file metadata. It must pass credentials from the
    router deployment, not just custom_llm_provider and file_id.

    Regression test for: managed_files.py:919 calling afile_retrieve without api_key/api_base.
    """
    managed_files = _make_managed_files_instance()
    batch_response = _make_batch_response(
        model_id="model-deploy-xyz",
        model_name="azure/gpt-4",
        output_file_id="file-output-abc",
    )
    user_api_key_dict = _make_user_api_key_dict()

    mock_credentials = {
        "api_key": "test-azure-key",
        "api_base": "https://my-azure.openai.azure.com/",
        "api_version": "2025-03-01-preview",
        "custom_llm_provider": "azure",
    }

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value=mock_credentials
    )

    mock_afile_retrieve = AsyncMock(return_value=_make_file_object("file-output-abc"))

    with patch(
        "litellm.afile_retrieve", mock_afile_retrieve
    ), patch(
        "litellm.proxy.proxy_server.llm_router", mock_router
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

        mock_afile_retrieve.assert_called()
        call_kwargs = mock_afile_retrieve.call_args

        assert call_kwargs.kwargs.get("api_key") == "test-azure-key", (
            f"afile_retrieve must receive api_key from router credentials. "
            f"Got kwargs: {call_kwargs.kwargs}"
        )
        assert call_kwargs.kwargs.get("api_base") == "https://my-azure.openai.azure.com/", (
            f"afile_retrieve must receive api_base from router credentials. "
            f"Got kwargs: {call_kwargs.kwargs}"
        )


@pytest.mark.asyncio
async def test_should_fallback_when_no_router():
    """
    When llm_router is not available, afile_retrieve should still be called
    with the fallback behavior (custom_llm_provider extracted from model_name).
    """
    managed_files = _make_managed_files_instance()
    batch_response = _make_batch_response(
        model_id="model-deploy-xyz",
        model_name="azure/gpt-4",
        output_file_id="file-output-abc",
    )
    user_api_key_dict = _make_user_api_key_dict()

    mock_afile_retrieve = AsyncMock(return_value=_make_file_object("file-output-abc"))

    with patch(
        "litellm.afile_retrieve", mock_afile_retrieve
    ), patch(
        "litellm.proxy.proxy_server.llm_router", None
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

        mock_afile_retrieve.assert_called()
        call_kwargs = mock_afile_retrieve.call_args
        assert call_kwargs.kwargs.get("custom_llm_provider") == "azure"
        assert call_kwargs.kwargs.get("file_id") == "file-output-abc"
