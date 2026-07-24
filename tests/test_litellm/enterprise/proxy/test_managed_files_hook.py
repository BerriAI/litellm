"""
Tests for enterprise/litellm_enterprise/proxy/hooks/managed_files.py

Regression test for afile_retrieve called without credentials in
async_post_call_success_hook when processing completed batch responses.
"""

import pytest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import types

# The hook imports litellm.proxy.proxy_server at call time; unit tests only need llm_router.
if "litellm.proxy.proxy_server" not in sys.modules:
    _proxy_server_stub = types.ModuleType("litellm.proxy.proxy_server")
    _proxy_server_stub.llm_router = None
    sys.modules["litellm.proxy.proxy_server"] = _proxy_server_stub

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

    with (
        patch("litellm.afile_retrieve", mock_afile_retrieve),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
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
        assert (
            call_kwargs.kwargs.get("api_base") == "https://my-azure.openai.azure.com/"
        ), (
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

    with (
        patch("litellm.afile_retrieve", mock_afile_retrieve),
        patch("litellm.proxy.proxy_server.llm_router", None),
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


@pytest.mark.asyncio
async def test_should_not_double_wrap_already_unified_output_file_id():
    """After ensure_batch_response_managed_file_ids, retrieve must not re-wrap
    output_file_id or store a nested unified id as the provider mapping."""
    import base64

    managed_files = _make_managed_files_instance()
    provider_file_id = "file-WXWt9R4LzmU5WpeKzjCfLR"
    model_id = "openai/openai/gpt-5.5-batch"
    already_unified = managed_files.get_unified_output_file_id(
        output_file_id=provider_file_id,
        model_id=model_id,
        model_name="openai/openai/gpt-5.5-batch",
    )

    batch_response = _make_batch_response(
        model_id=model_id,
        model_name="openai/openai/gpt-5.5-batch",
        output_file_id=already_unified,
    )
    user_api_key_dict = _make_user_api_key_dict()

    mock_credentials = {
        "api_key": "test-key",
        "api_base": "https://api.openai.com/v1",
        "custom_llm_provider": "openai",
    }
    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value=mock_credentials
    )
    mock_afile_retrieve = AsyncMock(return_value=_make_file_object(provider_file_id))

    with (
        patch("litellm.afile_retrieve", mock_afile_retrieve),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

    assert batch_response.output_file_id == already_unified
    mock_afile_retrieve.assert_called_once()
    assert mock_afile_retrieve.call_args.kwargs["file_id"] == provider_file_id
    managed_files.store_unified_file_id.assert_awaited_once()
    assert managed_files.store_unified_file_id.await_args.kwargs["model_mappings"] == {
        model_id: provider_file_id
    }

    decoded = base64.urlsafe_b64decode(
        already_unified + "=" * (-len(already_unified) % 4)
    ).decode()
    assert decoded.count(f"llm_output_file_id,{provider_file_id}") == 1


@pytest.mark.asyncio
async def test_should_unwrap_nested_unified_output_file_id():
    """Nested managed output IDs must unwrap to the raw provider file id.

    Regression for #33988: repeated retrieve can store a managed unified id
    inside another managed unified id. model_mappings must still point at the
    raw provider id (file-* / gs://), and afile_retrieve must receive that raw id.
    """
    managed_files = _make_managed_files_instance()

    provider_file_id = "file-WXWt9R4LzmU5WpeKzjCfLR"
    model_id = "openai/openai/gpt-5.5-batch"
    model_name = "openai/openai/gpt-5.5-batch"
    inner_unified = managed_files.get_unified_output_file_id(
        output_file_id=provider_file_id,
        model_id=model_id,
        model_name=model_name,
    )
    # Simulate a buggy prior retrieve that wrapped the unified id again.
    nested_unified = managed_files.get_unified_output_file_id(
        output_file_id=inner_unified,
        model_id=model_id,
        model_name=model_name,
    )

    batch_response = _make_batch_response(
        model_id=model_id,
        model_name=model_name,
        output_file_id=nested_unified,
    )
    user_api_key_dict = _make_user_api_key_dict()

    mock_credentials = {
        "api_key": "test-key",
        "api_base": "https://api.openai.com/v1",
        "custom_llm_provider": "openai",
    }
    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value=mock_credentials
    )
    mock_afile_retrieve = AsyncMock(return_value=_make_file_object(provider_file_id))

    with (
        patch("litellm.afile_retrieve", mock_afile_retrieve),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

    # Keep the outermost id; do not wrap a third time.
    assert batch_response.output_file_id == nested_unified
    mock_afile_retrieve.assert_called_once()
    assert mock_afile_retrieve.call_args.kwargs["file_id"] == provider_file_id
    managed_files.store_unified_file_id.assert_awaited_once()
    assert managed_files.store_unified_file_id.await_args.kwargs["model_mappings"] == {
        model_id: provider_file_id
    }


@pytest.mark.asyncio
async def test_should_skip_non_file_unified_id_on_output_file_id():

    """Batch-style unified ids lack llm_output_file_id; must not IndexError or re-wrap."""
    import base64

    managed_files = _make_managed_files_instance()
    batch_unified = (
        base64.urlsafe_b64encode(
            b"litellm_proxy;model_id:openai/openai/gpt-5.5-batch;llm_batch_id:batch_abc"
        )
        .decode()
        .rstrip("=")
    )

    batch_response = _make_batch_response(
        model_id="openai/openai/gpt-5.5-batch",
        model_name="openai/openai/gpt-5.5-batch",
        output_file_id=batch_unified,
    )
    user_api_key_dict = _make_user_api_key_dict()

    with patch("litellm.afile_retrieve", AsyncMock()) as mock_afile_retrieve:
        await managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch_response,
        )

    assert batch_response.output_file_id == batch_unified
    mock_afile_retrieve.assert_not_called()
    managed_files.store_unified_file_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_afile_content_passes_trusted_model_credentials_to_router():
    """
    afile_content must hand the deployment's credential snapshot to the router
    call as an immutable server-side mapping. Cloud-storage providers (Bedrock
    S3) validate file ids against the bucket in that snapshot, so without it
    unified-id content retrieval only works when AWS_S3_BUCKET_NAME is set.
    """
    from types import MappingProxyType

    managed_files = _make_managed_files_instance()
    unified_file_id = "unified-file-id"
    s3_uri = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    managed_files.get_model_file_id_mapping = AsyncMock(
        return_value={unified_file_id: {"model-123": s3_uri}}
    )

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value={
            "custom_llm_provider": "bedrock",
            "s3_bucket_name": "my-bucket",
            "aws_region_name": "us-west-2",
        }
    )
    mock_router.afile_content = AsyncMock(return_value=MagicMock())

    await managed_files.afile_content(
        file_id=unified_file_id,
        litellm_parent_otel_span=None,
        llm_router=mock_router,
    )

    call_kwargs = mock_router.afile_content.call_args.kwargs
    assert call_kwargs["model"] == "model-123"
    assert call_kwargs["file_id"] == s3_uri
    trusted_credentials = call_kwargs["_litellm_internal_model_credentials"]
    assert isinstance(trusted_credentials, MappingProxyType)
    assert trusted_credentials["s3_bucket_name"] == "my-bucket"


@pytest.mark.asyncio
async def test_afile_content_bedrock_unified_id_end_to_end(monkeypatch):
    """
    Proxy repro for Bedrock batch output retrieval: a unified file id that
    resolves to an s3:// output object must be fetched via a SigV4-signed S3
    GET using the deployment's s3_bucket_name (no AWS_S3_BUCKET_NAME env).

    Regression test for "BedrockFilesConfig does not support file content
    retrieval" raised on this path.
    """
    import httpx
    import respx

    import litellm
    from litellm import Router

    monkeypatch.delenv("AWS_S3_BUCKET_NAME", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()

    router = Router(
        model_list=[
            {
                "model_name": "bedrock-claude",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "aws_access_key_id": "AKIAEXAMPLE",
                    "aws_secret_access_key": "secret",
                    "aws_region_name": "us-west-2",
                    "s3_bucket_name": "my-bucket",
                },
                "model_info": {"id": "model-123"},
            }
        ]
    )

    managed_files = _make_managed_files_instance()
    unified_file_id = "unified-file-id"
    s3_uri = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    managed_files.get_model_file_id_mapping = AsyncMock(
        return_value={unified_file_id: {"model-123": s3_uri}}
    )

    expected_url = "https://s3.us-west-2.amazonaws.com/my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    with respx.mock:
        route = respx.get(expected_url).mock(
            return_value=httpx.Response(200, content=b'{"recordId": "x"}')
        )

        response = await managed_files.afile_content(
            file_id=unified_file_id,
            litellm_parent_otel_span=None,
            llm_router=router,
        )

    assert route.called
    assert (
        route.calls[0].request.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    )
    assert response.content == b'{"recordId": "x"}'


@pytest.mark.asyncio
async def test_afile_content_error_reports_unified_id_not_provider_uri():
    """When every model attempt fails, the error must name the caller's unified
    file id, never the resolved internal s3:// URI (no internal-path leak)."""
    managed_files = _make_managed_files_instance()
    unified_file_id = "litellm_proxy_unified_id_abc"
    s3_uri = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    managed_files.get_model_file_id_mapping = AsyncMock(
        return_value={unified_file_id: {"model-123": s3_uri}}
    )

    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(return_value=None)
    mock_router.afile_content = AsyncMock(side_effect=Exception("deployment failed"))

    with pytest.raises(Exception) as exc_info:
        await managed_files.afile_content(
            file_id=unified_file_id,
            litellm_parent_otel_span=None,
            llm_router=mock_router,
        )

    message = str(exc_info.value)
    assert unified_file_id in message
    assert s3_uri not in message
