import json
import os
import sys
from typing import List
from unittest.mock import ANY, AsyncMock

import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm.files.types import FileContentStreamingResult
from litellm.proxy._types import LiteLLM_UserTableFiltered, UserAPIKeyAuth
from litellm.proxy.hooks import get_proxy_hook
from litellm.proxy.management_endpoints.internal_user_endpoints import ui_view_users
from litellm.proxy.openai_files_endpoints.file_content_streaming_handler import (
    FileContentStreamingHandler,
)
from litellm.proxy.proxy_server import app
from litellm.types.llms.openai import HttpxBinaryResponseContent, OpenAIFileObject

client = TestClient(app)
from litellm.caching.caching import DualCache
from litellm.proxy.proxy_server import hash_token
from litellm.proxy.utils import ProxyLogging


@pytest.fixture
def llm_router() -> Router:
    llm_router = Router(
        model_list=[
            {
                "model_name": "azure-gpt-3-5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": "AZURE_AI_API_KEY",
                    "api_base": "AZURE_AI_API_BASE",
                    "api_version": "azure_api_version",
                },
                "model_info": {
                    "id": "azure-gpt-3-5-turbo-id",
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "openai_api_key",
                },
                "model_info": {
                    "id": "gpt-3.5-turbo-id",
                },
            },
            {
                "model_name": "gemini-2.0-flash",
                "litellm_params": {
                    "model": "gemini/gemini-2.0-flash",
                },
                "model_info": {
                    "id": "gemini-2.0-flash-id",
                },
            },
        ]
    )
    return llm_router


def setup_proxy_logging_object(monkeypatch, llm_router: Router) -> ProxyLogging:
    proxy_logging_object = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_object._add_proxy_hooks(llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_object
    )
    return proxy_logging_object


@pytest.mark.asyncio
async def test_stream_file_content_with_logging_closes_inner_iterator_on_early_exit():
    class MockStreamIterator:
        def __init__(self) -> None:
            self._chunks = iter([b"hello", b"world"])
            self.aclose = AsyncMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    stream_iterator = MockStreamIterator()
    proxy_logging_obj = AsyncMock()

    generator = FileContentStreamingHandler.stream_file_content_with_logging(
        stream_iterator=stream_iterator,
        proxy_logging_obj=proxy_logging_obj,
        user_api_key_dict=AsyncMock(),
        data={"litellm_call_id": "call-123"},
    )

    assert await generator.__anext__() == b"hello"

    await generator.aclose()

    stream_iterator.aclose.assert_awaited_once()
    proxy_logging_obj.update_request_status.assert_not_called()


def test_resolve_streaming_request_params_non_routed_returns_original_values():
    data = {"file_id": "file-abc123", "metadata": {"k": "v"}}

    (
        resolved_custom_llm_provider,
        resolved_file_id,
        resolved_streaming_data,
    ) = FileContentStreamingHandler.resolve_streaming_request_params(
        custom_llm_provider="openai",
        file_id="file-abc123",
        data=data,
        should_route=False,
        original_file_id=None,
        credentials=None,
    )

    assert resolved_custom_llm_provider == "openai"
    assert resolved_file_id == "file-abc123"
    assert resolved_streaming_data is data


def test_resolve_streaming_request_params_routed_uses_credentials_and_original_file_id():
    data = {
        "file_id": "file-encoded-123",
        "model": "azure-gpt-3-5-turbo",
        "metadata": {"k": "v"},
    }
    credentials = {
        "custom_llm_provider": "azure",
        "api_key": "azure-key",
        "api_base": "https://azure.example.com",
    }

    (
        resolved_custom_llm_provider,
        resolved_file_id,
        resolved_streaming_data,
    ) = FileContentStreamingHandler.resolve_streaming_request_params(
        custom_llm_provider="openai",
        file_id="file-encoded-123",
        data=data,
        should_route=True,
        original_file_id="file-original-123",
        credentials=credentials,
    )

    assert resolved_custom_llm_provider == "azure"
    assert resolved_file_id == "file-original-123"
    assert resolved_streaming_data["file_id"] == "file-original-123"
    assert resolved_streaming_data["api_key"] == "azure-key"
    assert resolved_streaming_data["api_base"] == "https://azure.example.com"
    assert "custom_llm_provider" not in resolved_streaming_data
    assert "model" not in resolved_streaming_data
    assert data["file_id"] == "file-encoded-123"
    assert data["model"] == "azure-gpt-3-5-turbo"


def test_resolve_streaming_request_params_routed_preserves_input_data_object():
    data = {
        "file_id": "file-encoded-123",
        "model": "openai/gpt-4o",
    }
    credentials = {
        "custom_llm_provider": "openai",
        "api_key": "sk-test",
    }

    (
        _resolved_custom_llm_provider,
        _resolved_file_id,
        resolved_streaming_data,
    ) = FileContentStreamingHandler.resolve_streaming_request_params(
        custom_llm_provider="openai",
        file_id="file-encoded-123",
        data=data,
        should_route=True,
        original_file_id=None,
        credentials=credentials,
    )

    assert resolved_streaming_data is not data
    assert data == {
        "file_id": "file-encoded-123",
        "model": "openai/gpt-4o",
    }


def test_invalid_purpose(mocker: MockerFixture, monkeypatch, llm_router: Router):
    """
    Asserts 'create_file' is called with the correct arguments
    """
    # Create a simple test file content
    test_file_content = b"test audio content"
    test_file = ("test.wav", test_file_content, "audio/wav")

    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "my-bad-purpose",
            # "target_model_names": ["azure-gpt-3-5-turbo", "gpt-3.5-turbo"],
            "target_model_names": "gpt-3-5-turbo",
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 400
    print(f"response: {response.json()}")
    assert "Invalid purpose: my-bad-purpose" in response.json()["error"]["message"]


def test_get_file_content_rejects_raw_cloud_storage_uri(llm_router: Router):
    """A raw s3:// file id must be rejected on the proxy content endpoint.

    Such an id is not a managed unified id, so it would otherwise skip the
    owner/team access check and let a caller read another tenant's batch output
    object by its key. Callers must use the managed unified file id.
    """
    from urllib.parse import quote

    s3_file_id = "s3://my-bucket/litellm-batch-outputs/job-123/input.jsonl.out"
    response = client.get(
        f"/v1/files/{quote(s3_file_id, safe='')}/content?provider=bedrock",
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 400
    assert "managed file id" in response.json()["error"]["message"].lower()


def test_mock_create_audio_file(mocker: MockerFixture, monkeypatch, llm_router: Router):
    """
    Asserts 'create_file' is called with the correct arguments
    """
    import litellm
    import litellm.proxy.proxy_server as ps
    from litellm import Router
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    # Mock create_file as an async function
    mock_create_file = mocker.patch(
        "litellm.files.main.create_file", new=mocker.AsyncMock()
    )

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )

    proxy_logging_obj._add_proxy_hooks(llm_router)

    # Add managed_files hook to ensure the test reaches the mocked function
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            # Handle both dict and object forms of create_file_request
            if isinstance(create_file_request, dict):
                file_data = create_file_request.get("file")
                purpose_data = create_file_request.get("purpose")
            else:
                file_data = create_file_request.file
                purpose_data = create_file_request.purpose

            # Call the mocked litellm.files.main.create_file to ensure asserts work
            await litellm.files.main.create_file(
                custom_llm_provider="azure",
                model="azure/chatgpt-v-2",
                api_key="AZURE_AI_API_KEY",
                file=file_data[1],
                purpose=purpose_data,
            )
            await litellm.files.main.create_file(
                custom_llm_provider="openai",
                model="openai/gpt-3.5-turbo",
                api_key="openai_api_key",
                file=file_data[1],
                purpose=purpose_data,
            )
            # Return a dummy response object as needed by the test
            from litellm.types.llms.openai import OpenAIFileObject

            return OpenAIFileObject(
                id="dummy-id",
                object="file",
                bytes=len(file_data[1]) if file_data else 0,
                created_at=1234567890,
                filename=file_data[0] if file_data else "test.wav",
                purpose=purpose_data,
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError("Not implemented for test")

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError("Not implemented for test")

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

    # Manually add the hook to the proxy_hook_mapping
    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()

    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    try:
        # Create a simple test file content
        test_file_content = b"test audio content"
        test_file = ("test.wav", test_file_content, "audio/wav")

        response = client.post(
            "/v1/files",
            files={"file": test_file},
            data={
                "purpose": "user_data",
                "target_model_names": "azure-gpt-3-5-turbo, gpt-3.5-turbo",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200

        # Get all calls made to create_file
        calls = mock_create_file.call_args_list

        # Check for Azure call
        azure_call_found = False
        for call in calls:
            kwargs = call.kwargs
            if (
                kwargs.get("custom_llm_provider") == "azure"
                and kwargs.get("model") == "azure/chatgpt-v-2"
                and kwargs.get("api_key") == "AZURE_AI_API_KEY"
            ):
                azure_call_found = True
                break
        assert (
            azure_call_found
        ), f"Azure call not found with expected parameters. Calls: {calls}"

        # Check for OpenAI call
        openai_call_found = False
        for call in calls:
            kwargs = call.kwargs
            if (
                kwargs.get("custom_llm_provider") == "openai"
                and kwargs.get("model") == "openai/gpt-3.5-turbo"
                and kwargs.get("api_key") == "openai_api_key"
            ):
                openai_call_found = True
                break
        assert openai_call_found, "OpenAI call not found with expected parameters"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_create_file_batch_streams_from_upload_spool(monkeypatch, llm_router: Router):
    """
    Batch uploads must be passed downstream as the upload's streamable file handle
    (Starlette's already-spooled file), not read into an in-memory bytes object, so
    the proxy never buffers the whole payload. Non-batch uploads keep the bytes path.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.openai_files_endpoints import files_endpoints as fe
    from litellm.types.llms.openai import OpenAIFileObject

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    captured: dict = {}

    async def fake_route_create_file(*, _create_file_request, **kwargs):
        file_elem = _create_file_request["file"][1]
        captured["file_elem"] = file_elem
        if hasattr(file_elem, "read") and hasattr(file_elem, "seek"):
            file_elem.seek(0)
            captured["streamed_content"] = file_elem.read()
        return OpenAIFileObject(
            id="dummy-id",
            object="file",
            bytes=0,
            created_at=1234567890,
            filename="batch.jsonl",
            purpose="batch",
            status="uploaded",
        )

    monkeypatch.setattr(fe, "route_create_file", fake_route_create_file)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    content = (
        b'{"custom_id":"r-0","method":"POST","url":"/v1/chat/completions",'
        b'"body":{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"hi"}]}}\n'
    )
    try:
        resp = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", content, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 200, resp.text
        file_elem = captured["file_elem"]
        assert not isinstance(
            file_elem, (bytes, bytearray)
        ), "batch upload must be a streamable handle, not in-memory bytes"
        assert hasattr(file_elem, "read") and hasattr(
            file_elem, "seek"
        ), "batch upload must be a seekable file handle"
        assert (
            captured["streamed_content"] == content
        ), "the handle must stream the uploaded bytes"

        captured.clear()
        resp = client.post(
            "/v1/files",
            files={"file": ("data.jsonl", content, "application/jsonl")},
            data={"purpose": "user_data"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 200, resp.text
        assert isinstance(
            captured["file_elem"], (bytes, bytearray)
        ), "non-batch upload must stay in-memory bytes"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.flaky(retries=3, delay=2)
def test_target_storage_invokes_storage_backend(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Ensure target_storage is parsed and invokes the storage backend service.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    async_mock = mocker.AsyncMock(
        return_value=OpenAIFileObject(
            id="file-test",
            object="file",
            purpose="user_data",
            created_at=0,
            bytes=3,
            filename="abc.txt",
            status="uploaded",
        )
    )
    mocker.patch(
        "litellm.proxy.openai_files_endpoints.storage_backend_service.StorageBackendFileService.upload_file_to_storage_backend",
        new=async_mock,
    )

    try:
        test_file_content = b"abc"
        test_file = ("abc.txt", test_file_content, "text/plain")

        response = client.post(
            "/v1/files",
            files={"file": test_file},
            data={
                "purpose": "user_data",
                "target_storage": "azure_storage",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200, response.text
        async_mock.assert_awaited_once()
        called_kwargs = async_mock.call_args.kwargs
        assert called_kwargs["target_storage"] == "azure_storage"
        assert called_kwargs["target_model_names"] == []
        assert called_kwargs["purpose"] == "user_data"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.flaky(retries=3, delay=2)
def test_target_storage_with_target_models(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Ensure target_storage and target_model_names are parsed and passed through.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    async_mock = mocker.AsyncMock(
        return_value=OpenAIFileObject(
            id="file-test",
            object="file",
            purpose="user_data",
            created_at=0,
            bytes=3,
            filename="abc.txt",
            status="uploaded",
        )
    )
    mocker.patch(
        "litellm.proxy.openai_files_endpoints.storage_backend_service.StorageBackendFileService.upload_file_to_storage_backend",
        new=async_mock,
    )

    try:
        test_file_content = b"abc"
        test_file = ("abc.txt", test_file_content, "text/plain")

        response = client.post(
            "/v1/files",
            files={"file": test_file},
            data={
                "purpose": "user_data",
                "target_storage": "azure_storage",
                "target_model_names": "gemini-2.0-flash",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        assert response.status_code == 200, response.text
        async_mock.assert_awaited_once()
        called_kwargs = async_mock.call_args.kwargs
        assert called_kwargs["target_storage"] == "azure_storage"
        assert called_kwargs["target_model_names"] == ["gemini-2.0-flash"]
        assert called_kwargs["purpose"] == "user_data"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


@pytest.mark.skip(reason="mock respx fails on ci/cd - unclear why")
def test_create_file_and_call_chat_completion_e2e(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    1. Create a file
    2. Call a chat completion with the file
    3. Assert the file is used in the chat completion
    """
    # Create and enable respx mock instance
    mock = respx.mock()
    mock.start()
    try:
        from litellm.types.llms.openai import OpenAIFileObject

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
        proxy_logging_object = setup_proxy_logging_object(monkeypatch, llm_router)

        # Create a simple test file content
        test_file_content = b"test audio content"
        test_file = ("test.wav", test_file_content, "audio/wav")

        # Mock the file creation response
        mock_file_response = OpenAIFileObject(
            id="test-file-id",
            object="file",
            bytes=123,
            created_at=1234567890,
            filename="test.wav",
            purpose="user_data",
            status="uploaded",
        )
        mock_file_response._hidden_params = {"model_id": "gemini-2.0-flash-id"}
        mocker.patch.object(llm_router, "acreate_file", return_value=mock_file_response)

        # Mock the Gemini API call using respx
        mock_gemini_response = {
            "candidates": [
                {"content": {"parts": [{"text": "This is a test audio file"}]}}
            ]
        }

        # Mock the Gemini API endpoint with a more flexible pattern
        gemini_route = mock.post(
            url__regex=r".*generativelanguage\.googleapis\.com.*"
        ).mock(
            return_value=respx.MockResponse(status_code=200, json=mock_gemini_response),
        )

        # Print updated mock setup
        print("\nAfter Adding Gemini Route:")
        print("==========================")
        print(f"Number of mocked routes: {len(mock.routes)}")
        for route in mock.routes:
            print(f"Mocked Route: {route}")
            print(f"Pattern: {route.pattern}")

        ## CREATE FILE
        file = client.post(
            "/v1/files",
            files={"file": test_file},
            data={
                "purpose": "user_data",
                "target_model_names": "gemini-2.0-flash, gpt-3.5-turbo",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        print("\nAfter File Creation:")
        print("====================")
        print(f"File creation status: {file.status_code}")
        print(f"Recorded calls so far: {len(mock.calls)}")
        for call in mock.calls:
            print(f"Call made to: {call.request.method} {call.request.url}")

        assert file.status_code == 200
        assert file.json()["id"] != "test-file-id"  # unified file id used

        ## USE FILE IN CHAT COMPLETION
        try:
            completion = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-2.0-flash",
                    "modalities": ["text", "audio"],
                    "audio": {"voice": "alloy", "format": "wav"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is in this recording?"},
                                {
                                    "type": "file",
                                    "file": {
                                        "file_id": file.json()["id"],
                                        "filename": "my-test-name",
                                        "format": "audio/wav",
                                    },
                                },
                            ],
                        },
                    ],
                    "drop_params": True,
                },
                headers={"Authorization": "Bearer test-key"},
            )
        except Exception as e:
            print(f"error: {e}")

        print("\nError occurred during chat completion:")
        print("=====================================")
        print("\nFinal Mock State:")
        print("=================")
        print(f"Total mocked routes: {len(mock.routes)}")
        for route in mock.routes:
            print(f"\nMocked Route: {route}")
            print(f"  Called: {route.called}")

        print("\nActual Requests Made:")
        print("=====================")
        print(f"Total calls recorded: {len(mock.calls)}")
        for idx, call in enumerate(mock.calls):
            print(f"\nCall {idx + 1}:")
            print(f"  Method: {call.request.method}")
            print(f"  URL: {call.request.url}")
            print(f"  Headers: {dict(call.request.headers)}")
            try:
                print(f"  Body: {call.request.content.decode()}")
            except:
                print("  Body: <could not decode>")

        # Verify Gemini API was called
        assert gemini_route.called, "Gemini API was not called"

        # Print the call details
        print("\nGemini API Call Details:")
        print(f"URL: {gemini_route.calls.last.request.url}")
        print(f"Method: {gemini_route.calls.last.request.method}")
        print(f"Headers: {dict(gemini_route.calls.last.request.headers)}")
        print(f"Content: {gemini_route.calls.last.request.content.decode()}")
        print(f"Response: {gemini_route.calls.last.response.content.decode()}")

        assert "test-file-id" in gemini_route.calls.last.request.content.decode()
    finally:
        # Stop the mock
        mock.stop()


@pytest.mark.skip(reason="function migrated to litellm/proxy/hooks/managed_files.py")
def test_create_file_for_each_model(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that create_file_for_each_model creates files for each target model and returns a unified file ID
    """
    import asyncio

    from litellm import CreateFileRequest
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.openai_files_endpoints.files_endpoints import (
        create_file_for_each_model,
    )
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.llms.openai import OpenAIFileObject, OpenAIFilesPurpose

    # Setup proxy logging
    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    # Mock user API key dict
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
        team_id="test-team",
        team_alias="test-team-alias",
        parent_otel_span=None,
    )

    # Create test file request
    test_file_content = b"test file content"
    test_file = ("test.txt", test_file_content, "text/plain")
    _create_file_request = CreateFileRequest(file=test_file, purpose="user_data")

    # Mock the router's acreate_file method
    mock_file_response = OpenAIFileObject(
        id="test-file-id",
        object="file",
        bytes=123,
        created_at=1234567890,
        filename="test.txt",
        purpose="user_data",
        status="uploaded",
    )
    mock_file_response._hidden_params = {"model_id": "test-model-id"}
    mocker.patch.object(llm_router, "acreate_file", return_value=mock_file_response)

    # Call the function
    target_model_names_list = ["azure-gpt-3-5-turbo", "gpt-3.5-turbo"]
    response = asyncio.run(
        create_file_for_each_model(
            llm_router=llm_router,
            _create_file_request=_create_file_request,
            target_model_names_list=target_model_names_list,
            purpose="user_data",
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=user_api_key_dict,
        )
    )

    # Verify the response
    assert isinstance(response, OpenAIFileObject)
    assert response.id is not None
    assert response.purpose == "user_data"
    assert response.filename == "test.txt"

    # Verify acreate_file was called for each model
    assert llm_router.acreate_file.call_count == len(target_model_names_list)

    # Get all calls made to acreate_file
    calls = llm_router.acreate_file.call_args_list

    # Verify Azure call
    azure_call_found = False
    for call in calls:
        kwargs = call.kwargs
        if (
            kwargs.get("model") == "azure-gpt-3-5-turbo"
            and kwargs.get("file") == test_file
            and kwargs.get("purpose") == "user_data"
        ):
            azure_call_found = True
            break
    assert azure_call_found, "Azure call not found with expected parameters"

    # Verify OpenAI call
    openai_call_found = False
    for call in calls:
        kwargs = call.kwargs
        if (
            kwargs.get("model") == "gpt-3.5-turbo"
            and kwargs.get("file") == test_file
            and kwargs.get("purpose") == "user_data"
        ):
            openai_call_found = True
            break
    assert openai_call_found, "OpenAI call not found with expected parameters"


def test_create_file_with_expires_after(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that expires_after is properly parsed and passed through when creating a file
    """
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.types.llms.openai import OpenAIFileObject

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            # Verify expires_after is in the request
            if isinstance(create_file_request, dict):
                expires_after = create_file_request.get("expires_after")
            else:
                expires_after = getattr(create_file_request, "expires_after", None)

            # Verify expires_after was passed correctly
            assert expires_after is not None, "expires_after should be in the request"
            assert expires_after["anchor"] == "created_at"
            assert expires_after["seconds"] == 2592000

            # Return a dummy response
            return OpenAIFileObject(
                id="file-abc123",
                object="file",
                bytes=100,
                created_at=1234567890,
                filename="mydata.jsonl",
                purpose="fine-tune",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError("Not implemented for test")

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError("Not implemented for test")

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    # Create test file content
    test_file_content = b'{"prompt": "Hello", "completion": "Hi"}'
    test_file = ("mydata.jsonl", test_file_content, "application/json")

    # Test with expires_after
    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "fine-tune",
            "target_model_names": "gpt-3.5-turbo",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "2592000",  # 30 days
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "file-abc123"
    assert result["purpose"] == "fine-tune"


def test_create_file_with_expires_after_missing_anchor(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that an error is returned when expires_after[anchor] is missing
    """
    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    test_file_content = b'{"prompt": "Hello", "completion": "Hi"}'
    test_file = ("mydata.jsonl", test_file_content, "application/json")

    # Test with only expires_after[seconds], missing anchor
    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "fine-tune",
            "expires_after[seconds]": "2592000",
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 400
    error_detail = response.json()
    assert (
        "expires_after" in error_detail["error"]["message"].lower()
        or "both" in error_detail["error"]["message"].lower()
    )


def test_create_file_with_expires_after_missing_seconds(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that an error is returned when expires_after[seconds] is missing
    """
    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    test_file_content = b'{"prompt": "Hello", "completion": "Hi"}'
    test_file = ("mydata.jsonl", test_file_content, "application/json")

    # Test with only expires_after[anchor], missing seconds
    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "fine-tune",
            "expires_after[anchor]": "created_at",
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 400
    error_detail = response.json()
    assert (
        "expires_after" in error_detail["error"]["message"].lower()
        or "both" in error_detail["error"]["message"].lower()
    )


def test_create_file_with_expires_after_valid_values(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that expires_after works with valid anchor and seconds values
    """
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.types.llms.openai import OpenAIFileObject

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            # Verify expires_after is in the request
            if isinstance(create_file_request, dict):
                expires_after = create_file_request.get("expires_after")
            else:
                expires_after = getattr(create_file_request, "expires_after", None)

            # Verify expires_after was passed correctly
            assert expires_after is not None, "expires_after should be in the request"
            assert expires_after["anchor"] == "created_at"
            assert expires_after["seconds"] == 3600

            return OpenAIFileObject(
                id="file-abc123",
                object="file",
                bytes=100,
                created_at=1234567890,
                filename="mydata.jsonl",
                purpose="fine-tune",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError("Not implemented for test")

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError("Not implemented for test")

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    test_file_content = b'{"prompt": "Hello", "completion": "Hi"}'
    test_file = ("mydata.jsonl", test_file_content, "application/json")

    # Test with valid expires_after values
    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "fine-tune",
            "target_model_names": "gpt-3.5-turbo",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "3600",  # Minimum valid value (1 hour)
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "file-abc123"
    assert result["purpose"] == "fine-tune"


def test_create_file_without_expires_after(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that file creation works normally without expires_after
    """
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.types.llms.openai import OpenAIFileObject

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            # Verify expires_after is None when not provided
            if isinstance(create_file_request, dict):
                expires_after = create_file_request.get("expires_after")
            else:
                expires_after = getattr(create_file_request, "expires_after", None)

            # expires_after should be None when not provided
            assert (
                expires_after is None
            ), "expires_after should be None when not provided"

            return OpenAIFileObject(
                id="file-abc123",
                object="file",
                bytes=100,
                created_at=1234567890,
                filename="mydata.jsonl",
                purpose="fine-tune",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError("Not implemented for test")

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError("Not implemented for test")

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    test_file_content = b'{"prompt": "Hello", "completion": "Hi"}'
    test_file = ("mydata.jsonl", test_file_content, "application/json")

    # Test without expires_after
    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "fine-tune",
            "target_model_names": "gpt-3.5-turbo",
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "file-abc123"
    assert result["purpose"] == "fine-tune"


def test_managed_files_with_loadbalancing(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that managed files work with loadbalancing when both target_model_names
    and enable_loadbalancing_on_batch_endpoints are enabled.

    This ensures that the priority order is correct:
    - managed files should take precedence over deprecated loadbalancing
    - managed files internally use llm_router.acreate_file() which provides loadbalancing
    """
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.types.llms.openai import OpenAIFileObject

    # Enable loadbalancing on batch endpoints
    monkeypatch.setattr("litellm.enable_loadbalancing_on_batch_endpoints", True)

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)

    # Track calls to verify loadbalancing through router
    router_acreate_file_calls = []

    class ManagedFilesWithLoadbalancing(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            # Verify we receive the target model names
            assert (
                len(target_model_names_list) > 0
            ), "Should have target_model_names_list"

            # Simulate what managed files does - call llm_router.acreate_file for each model
            # This is where loadbalancing happens internally
            for model in target_model_names_list:
                router_acreate_file_calls.append({"model": model, "via_router": True})

            # Return a managed file ID (base64 encoded)
            return OpenAIFileObject(
                id="litellm_managed_file_abc123",
                object="file",
                bytes=100,
                created_at=1234567890,
                filename="batch_data.jsonl",
                purpose="batch",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError("Not implemented for test")

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError("Not implemented for test")

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = (
        ManagedFilesWithLoadbalancing()
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    # Override auth to avoid dependence on shared proxy state in parallel CI
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    try:
        # Create batch file content
        test_file_content = b'{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}}'
        test_file = ("batch_data.jsonl", test_file_content, "application/jsonl")

        # Make request with both target_model_names AND enable_loadbalancing_on_batch_endpoints
        response = client.post(
            "/v1/files",
            files={"file": test_file},
            data={
                "purpose": "batch",
                "target_model_names": "azure-gpt-3-5-turbo,gpt-3.5-turbo",  # Multiple models
            },
            headers={"Authorization": "Bearer test-key"},
        )

        # Verify success
        assert response.status_code == 200, response.text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
    result = response.json()
    assert result["id"] == "litellm_managed_file_abc123"
    assert result["purpose"] == "batch"

    # Verify that managed files was called (via router for loadbalancing)
    # This proves that managed files took precedence over deprecated loadbalancing
    assert (
        len(router_acreate_file_calls) == 2
    ), "Should have called router for both models"
    assert router_acreate_file_calls[0]["model"] == "azure-gpt-3-5-turbo"
    assert router_acreate_file_calls[1]["model"] == "gpt-3.5-turbo"
    assert all(
        call["via_router"] for call in router_acreate_file_calls
    ), "All calls should go through router"


def test_create_file_with_nested_litellm_metadata(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that nested litellm_metadata is correctly parsed from form data in bracket notation.

    Regression test for: litellm_metadata[spend_logs_metadata][owner] format should be
    correctly parsed into nested dictionary structure.
    """
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.types.llms.openai import OpenAIFileObject

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)

    captured_litellm_metadata = {}

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            # Capture litellm_metadata for verification
            if isinstance(create_file_request, dict):
                captured_litellm_metadata.update(
                    create_file_request.get("litellm_metadata", {})
                )
            else:
                captured_litellm_metadata.update(
                    getattr(create_file_request, "litellm_metadata", {})
                )

            return OpenAIFileObject(
                id="file-test-123",
                object="file",
                bytes=100,
                created_at=1234567890,
                filename="test.jsonl",
                purpose="fine-tune",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError("Not implemented for test")

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError("Not implemented for test")

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    test_file_content = b'{"prompt": "Hello", "completion": "Hi"}'
    test_file = ("test.jsonl", test_file_content, "application/jsonl")

    # Test with nested litellm_metadata in bracket notation
    response = client.post(
        "/v1/files",
        files={"file": test_file},
        data={
            "purpose": "fine-tune",
            "target_model_names": "gpt-3.5-turbo",
            "litellm_metadata[spend_logs_metadata][owner]": "john_doe",
            "litellm_metadata[spend_logs_metadata][team]": "engineering",
            "litellm_metadata[environment]": "prod",
        },
        headers={"Authorization": "Bearer test-key"},
    )

    # Verify success
    assert response.status_code == 200
    result = response.json()
    assert result["id"] == "file-test-123"

    # Verify nested metadata was correctly parsed.
    # Note: caller-supplied `tags` is stripped by default; test removed
    # to keep the parsing test focused on parser correctness.
    assert "spend_logs_metadata" in captured_litellm_metadata
    assert captured_litellm_metadata["spend_logs_metadata"]["owner"] == "john_doe"
    assert captured_litellm_metadata["spend_logs_metadata"]["team"] == "engineering"
    assert captured_litellm_metadata["environment"] == "prod"


def test_create_file_with_deep_nested_litellm_metadata(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Test that deeply nested litellm_metadata is correctly parsed from form data.

    Regression test for: litellm_metadata[a][b][c] format should be correctly parsed.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.proxy._types import LitellmUserRoles
    from litellm.types.llms.openai import OpenAIFileObject

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    proxy_logging_obj._add_proxy_hooks(llm_router)

    captured_litellm_metadata = {}

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            if isinstance(create_file_request, dict):
                captured_litellm_metadata.update(
                    create_file_request.get("litellm_metadata", {})
                )
            else:
                captured_litellm_metadata.update(
                    getattr(create_file_request, "litellm_metadata", {})
                )

            return OpenAIFileObject(
                id="file-test-456",
                object="file",
                bytes=50,
                created_at=1234567890,
                filename="nested.jsonl",
                purpose="batch",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError("Not implemented for test")

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError("Not implemented for test")

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError("Not implemented for test")

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    try:
        test_file_content = b'{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo"}}'
        test_file = ("nested.jsonl", test_file_content, "application/jsonl")

        # Test with deeply nested metadata
        response = client.post(
            "/v1/files",
            files={"file": test_file},
            data={
                "purpose": "batch",
                "target_model_names": "gpt-3.5-turbo",
                "litellm_metadata[config][database][host]": "localhost",
                "litellm_metadata[config][database][port]": "5432",
                "litellm_metadata[config][cache][enabled]": "true",
            },
            headers={"Authorization": "Bearer test-key"},
        )

        # Verify success
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["id"] == "file-test-456"

        # Verify deeply nested metadata was correctly parsed
        assert "config" in captured_litellm_metadata
        assert "database" in captured_litellm_metadata["config"]
        assert captured_litellm_metadata["config"]["database"]["host"] == "localhost"
        assert captured_litellm_metadata["config"]["database"]["port"] == "5432"
        assert "cache" in captured_litellm_metadata["config"]
        assert captured_litellm_metadata["config"]["cache"]["enabled"] == "true"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)


# ---------------------------------------------------------------------------
# Team-level enforced_file_expires_after tests
# ---------------------------------------------------------------------------


def _make_capturing_managed_files():
    """Create a DummyManagedFiles that captures the expires_after from the request."""
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints

    captured = {}

    class CapturingManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            if isinstance(create_file_request, dict):
                captured["expires_after"] = create_file_request.get("expires_after")
            else:
                captured["expires_after"] = getattr(
                    create_file_request, "expires_after", None
                )
            return OpenAIFileObject(
                id="file-abc123",
                object="file",
                bytes=100,
                created_at=1234567890,
                filename="mydata.jsonl",
                purpose="batch",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

    return CapturingManagedFiles(), captured


def _post_file_with_team_metadata(
    monkeypatch,
    llm_router: Router,
    team_metadata: dict,
    form_data: dict,
):
    """POST /v1/files with given team_metadata, return captured expires_after."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    dummy, captured = _make_capturing_managed_files()
    proxy_logging_obj.proxy_hook_mapping["managed_files"] = dummy
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    user_key = UserAPIKeyAuth(api_key="test-key", team_metadata=team_metadata)
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    test_file = ("mydata.jsonl", b'{"prompt": "Hello"}', "application/json")
    try:
        response = client.post(
            "/v1/files",
            files={"file": test_file},
            data=form_data,
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()

    return captured["expires_after"]


def test_file_team_override_overrides_caller(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Team enforced_file_expires_after wins over caller-provided value."""
    expires_after = _post_file_with_team_metadata(
        monkeypatch,
        llm_router,
        team_metadata={
            "enforced_file_expires_after": {
                "anchor": "created_at",
                "seconds": 3600,
            }
        },
        form_data={
            "purpose": "batch",
            "target_model_names": "gpt-3.5-turbo",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "86400",
        },
    )
    assert expires_after["anchor"] == "created_at"
    assert expires_after["seconds"] == 3600


def test_file_no_team_setting_preserves_caller(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """No team setting = caller-provided expires_after passes through."""
    expires_after = _post_file_with_team_metadata(
        monkeypatch,
        llm_router,
        team_metadata={},
        form_data={
            "purpose": "batch",
            "target_model_names": "gpt-3.5-turbo",
            "expires_after[anchor]": "created_at",
            "expires_after[seconds]": "86400",
        },
    )
    assert expires_after["anchor"] == "created_at"
    assert expires_after["seconds"] == 86400


def test_file_team_injects_when_caller_sends_nothing(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Team enforcement applies even when caller sends no expiry."""
    expires_after = _post_file_with_team_metadata(
        monkeypatch,
        llm_router,
        team_metadata={
            "enforced_file_expires_after": {
                "anchor": "created_at",
                "seconds": 3600,
            }
        },
        form_data={
            "purpose": "batch",
            "target_model_names": "gpt-3.5-turbo",
        },
    )
    assert expires_after["anchor"] == "created_at"
    assert expires_after["seconds"] == 3600


# ---------------------------------------------------------------------------
# Team-level enforced_file_expires_after validation error tests
# ---------------------------------------------------------------------------


def _post_file_raw(
    monkeypatch, llm_router: Router, team_metadata: dict, form_data: dict
):
    """POST /v1/files and return the raw response (no status assertion)."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    dummy, _ = _make_capturing_managed_files()
    proxy_logging_obj.proxy_hook_mapping["managed_files"] = dummy
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )

    user_key = UserAPIKeyAuth(api_key="test-key", team_metadata=team_metadata)
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    test_file = ("mydata.jsonl", b'{"prompt": "Hello"}', "application/json")
    try:
        response = client.post(
            "/v1/files",
            files={"file": test_file},
            data=form_data,
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.clear()

    return response


def test_file_missing_anchor_key_returns_500(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Missing 'anchor' key in team metadata returns 500."""
    response = _post_file_raw(
        monkeypatch,
        llm_router,
        team_metadata={
            "enforced_file_expires_after": {"seconds": 3600},
        },
        form_data={
            "purpose": "batch",
            "target_model_names": "gpt-3.5-turbo",
        },
    )
    assert response.status_code == 500
    assert "malformed" in response.json()["error"]["message"]


def test_file_missing_seconds_key_returns_500(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Missing 'seconds' key in team metadata returns 500."""
    response = _post_file_raw(
        monkeypatch,
        llm_router,
        team_metadata={
            "enforced_file_expires_after": {"anchor": "created_at"},
        },
        form_data={
            "purpose": "batch",
            "target_model_names": "gpt-3.5-turbo",
        },
    )
    assert response.status_code == 500
    assert "malformed" in response.json()["error"]["message"]


def test_file_invalid_anchor_returns_500(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Invalid anchor value in team metadata returns 500."""
    response = _post_file_raw(
        monkeypatch,
        llm_router,
        team_metadata={
            "enforced_file_expires_after": {
                "anchor": "updated_at",
                "seconds": 3600,
            },
        },
        form_data={
            "purpose": "batch",
            "target_model_names": "gpt-3.5-turbo",
        },
    )
    assert response.status_code == 500
    assert "created_at" in response.json()["error"]["message"]


def test_get_file_content_streams_openai_direct_path(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs = {}

    async def _mock_afile_content(**kwargs):
        captured_kwargs.update(kwargs)

        async def _stream():
            yield b"hello "
            yield b"world"

        return FileContentStreamingResult(
            stream_iterator=_stream(),
            headers={"content-length": "11"},
        )

    monkeypatch.setattr(litellm, "afile_content", _mock_afile_content)
    monkeypatch.setattr(
        "litellm.proxy.openai_files_endpoints.files_endpoints.handle_model_based_routing",
        lambda **kwargs: (False, None, None, None),
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = client.get(
            "/v1/files/file-abc123/content",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert response.content == b"hello world"
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers["content-length"] == "11"
    assert captured_kwargs["custom_llm_provider"] == "openai"
    assert captured_kwargs["file_id"] == "file-abc123"
    assert captured_kwargs["stream"] is True
    proxy_logging_obj.update_request_status.assert_awaited_once()
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_get_file_content_routed_provider_skips_streaming_when_resolved_provider_is_not_supported(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs = {}

    async def _mock_afile_content(**kwargs):
        captured_kwargs.update(kwargs)
        return HttpxBinaryResponseContent(
            response=httpx.Response(
                status_code=200,
                content=b"azure-bytes",
                headers={
                    "content-type": "application/octet-stream",
                    "content-length": "11",
                },
            )
        )

    mock_streaming_response = mocker.AsyncMock()

    monkeypatch.setattr(litellm, "afile_content", _mock_afile_content)
    monkeypatch.setattr(
        FileContentStreamingHandler,
        "get_streaming_file_content_response",
        mock_streaming_response,
    )
    monkeypatch.setattr(
        "litellm.proxy.openai_files_endpoints.files_endpoints.handle_model_based_routing",
        lambda **kwargs: (
            True,
            "azure-gpt-3-5-turbo",
            "file-original-123",
            {
                "custom_llm_provider": "azure",
                "api_key": "azure-key",
                "api_base": "https://azure.example.com",
            },
        ),
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = client.get(
            "/v1/files/file-abc123/content",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert response.content == b"azure-bytes"
    assert captured_kwargs["custom_llm_provider"] == "azure"
    assert captured_kwargs["file_id"] == "file-original-123"
    assert captured_kwargs["api_key"] == "azure-key"
    assert captured_kwargs["api_base"] == "https://azure.example.com"
    assert "stream" not in captured_kwargs
    mock_streaming_response.assert_not_awaited()
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_get_file_content_non_openai_provider_skips_streaming_handler(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs = {}

    async def _mock_afile_content(**kwargs):
        captured_kwargs.update(kwargs)
        return HttpxBinaryResponseContent(
            response=httpx.Response(
                status_code=200,
                content=b"azure-bytes",
                headers={
                    "content-type": "application/octet-stream",
                    "content-length": "11",
                },
            )
        )

    mock_streaming_response = mocker.AsyncMock()

    monkeypatch.setattr(litellm, "afile_content", _mock_afile_content)
    monkeypatch.setattr(
        FileContentStreamingHandler,
        "get_streaming_file_content_response",
        mock_streaming_response,
    )
    monkeypatch.setattr(
        "litellm.proxy.openai_files_endpoints.files_endpoints.handle_model_based_routing",
        lambda **kwargs: (False, None, None, None),
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = client.get(
            "/v1/files/file-abc123/content",
            headers={
                "Authorization": "Bearer test-key",
                "custom-llm-provider": "azure",
            },
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert response.content == b"azure-bytes"
    assert captured_kwargs["custom_llm_provider"] == "azure"
    assert "stream" not in captured_kwargs
    mock_streaming_response.assert_not_awaited()
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_require_managed_files_rejects_missing_target_model_names(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.require_managed_files", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    mock_acreate_file = mocker.patch("litellm.acreate_file", new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("test.txt", b"abc", "text/plain")},
            data={"purpose": "user_data"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        monkeypatch.setattr("litellm.require_managed_files", False)

    assert response.status_code == 400, response.text
    error_message = response.json()["error"]["message"]
    assert error_message.startswith("target_model_names is required")
    assert not error_message.startswith("{")
    mock_acreate_file.assert_not_called()


def test_require_managed_files_allows_managed_file_upload(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.require_managed_files", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            return OpenAIFileObject(
                id="litellm_managed_file_abc123",
                object="file",
                bytes=3,
                created_at=1234567890,
                filename="test.txt",
                purpose="user_data",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()

    mock_acreate_file = mocker.patch("litellm.acreate_file", new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("test.txt", b"abc", "text/plain")},
            data={
                "purpose": "user_data",
                "target_model_names": "gpt-3.5-turbo",
            },
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        monkeypatch.setattr("litellm.require_managed_files", False)

    assert response.status_code == 200, response.text
    assert response.json()["id"] == "litellm_managed_file_abc123"
    mock_acreate_file.assert_not_called()


def test_require_managed_files_rejects_model_param_bypass(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    Supplying model alongside target_model_names must not bypass managed files:
    route_create_file would otherwise take the model branch and call
    litellm.acreate_file directly instead of the managed-files hook.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.require_managed_files", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    mock_acreate_file = mocker.patch("litellm.acreate_file", new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("test.txt", b"abc", "text/plain")},
            data={
                "purpose": "user_data",
                "target_model_names": "gpt-3.5-turbo",
                "model": "gpt-3.5-turbo",
            },
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        monkeypatch.setattr("litellm.require_managed_files", False)

    assert response.status_code == 400, response.text
    error_message = response.json()["error"]["message"]
    assert error_message.startswith("model is not allowed")
    mock_acreate_file.assert_not_called()


def test_require_managed_files_accepts_target_model_names_bracket_form(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    OpenAI SDK sends list extra_body as target_model_names[] in multipart form.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.require_managed_files", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            assert target_model_names_list == ["gpt-3.5-turbo"]
            return OpenAIFileObject(
                id="litellm_managed_file_bracket",
                object="file",
                bytes=3,
                created_at=1234567890,
                filename="test.txt",
                purpose="user_data",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("test.txt", b"abc", "text/plain")},
            data={
                "purpose": "user_data",
                "target_model_names[]": "gpt-3.5-turbo",
            },
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        monkeypatch.setattr("litellm.require_managed_files", False)

    assert response.status_code == 200, response.text
    assert response.json()["id"] == "litellm_managed_file_bracket"


def test_require_managed_files_accepts_repeated_target_model_names_bracket_form(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """
    The OpenAI SDK serialises a list extra_body as repeated target_model_names[]
    fields. dict(form_data) keeps only the last one, so every value must survive.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.require_managed_files", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)

    received_target_model_names: List[str] = []

    class DummyManagedFiles(BaseFileEndpoints):
        async def acreate_file(
            self,
            llm_router,
            create_file_request,
            target_model_names_list,
            litellm_parent_otel_span,
            user_api_key_dict,
        ):
            received_target_model_names.extend(target_model_names_list)
            return OpenAIFileObject(
                id="litellm_managed_file_repeated",
                object="file",
                bytes=3,
                created_at=1234567890,
                filename="test.txt",
                purpose="user_data",
                status="uploaded",
            )

        async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router):
            raise NotImplementedError

        async def afile_list(self, purpose, litellm_parent_otel_span):
            raise NotImplementedError

        async def afile_delete(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

        async def afile_content(
            self, file_id, litellm_parent_otel_span, llm_router, **data
        ):
            raise NotImplementedError

    proxy_logging_obj.proxy_hook_mapping["managed_files"] = DummyManagedFiles()

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test-user"
    )

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("test.txt", b"abc", "text/plain")},
            data={
                "purpose": "user_data",
                "target_model_names[]": ["azure-gpt-3-5-turbo", "gpt-3.5-turbo"],
            },
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        monkeypatch.setattr("litellm.require_managed_files", False)

    assert response.status_code == 200, response.text
    assert response.json()["id"] == "litellm_managed_file_repeated"
    assert received_target_model_names == ["azure-gpt-3-5-turbo", "gpt-3.5-turbo"]


def test_list_files_resolves_wildcard_deployment_credentials(
    mocker: MockerFixture, monkeypatch
):
    """
    GET /v1/files?target_model_names=<model> must resolve the upstream api_key
    from the matching (wildcard) deployment. Regression for the path routing
    through llm_router.afile_list(model=...), which reached OpenAI without an
    api_key and failed with "api_key client option must be set".
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    wildcard_router = Router(
        model_list=[
            {
                "model_name": "*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "wildcard-openai-key",
                },
            },
        ]
    )

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, wildcard_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", wildcard_router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=[])
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_list(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(litellm, "afile_list", _mock_afile_list)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = client.get(
            "/v1/files?target_model_names=gpt-4o",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("api_key") == "wildcard-openai-key"
    assert captured_kwargs.get("custom_llm_provider") == "openai"
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_list_files_without_target_model_names_uses_team_openai_deployment(
    mocker: MockerFixture, monkeypatch
):
    """
    Plain GET /v1/files (no target_model_names) must resolve the upstream openai
    api_key from the team's openai deployment instead of falling through to a
    keyless OpenAI client. Regression for "api_key client option must be set".
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    wildcard_router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "team-openai-key",
                },
            },
        ]
    )

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, wildcard_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", wildcard_router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=[])
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_list(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(litellm, "afile_list", _mock_afile_list)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
        team_id="test-team",
        team_models=["openai/*"],
    )

    try:
        response = client.get(
            "/v1/files",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("api_key") == "team-openai-key"
    assert captured_kwargs.get("custom_llm_provider") == "openai"
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_list_files_restricted_team_does_not_leak_global_openai_credentials(
    mocker: MockerFixture, monkeypatch
):
    """
    A team whose allowlist only grants anthropic must NOT resolve a global
    openai deployment's api_key for plain GET /v1/files. Regression for the
    last-resort scan that ignored team access control.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "global-openai-key",
                },
            },
            {
                "model_name": "claude-opus-4-6",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-6",
                    "api_key": "anthropic-key",
                },
            },
        ]
    )

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=[])
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_list(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(litellm, "afile_list", _mock_afile_list)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
        team_id="anthropic-only-team",
        team_models=["claude-opus-4-6"],
    )

    try:
        response = client.get(
            "/v1/files",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("api_key") != "global-openai-key"


def test_list_files_prefers_team_byok_over_global_openai_deployment(
    mocker: MockerFixture, monkeypatch
):
    """
    When a team has its own BYOK openai deployment (model_info.team_id set), plain
    GET /v1/files must use the team's key, not a shared/global openai deployment.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "global-openai-key",
                },
            },
            {
                "model_name": "team-gpt-4o",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "team-byok-openai-key",
                },
                "model_info": {
                    "id": "team-byok-deployment-id",
                    "team_id": "test-team",
                    "team_public_model_name": "team-gpt-4o",
                },
            },
        ]
    )

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=[])
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_list(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(litellm, "afile_list", _mock_afile_list)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
        team_id="test-team",
        team_models=["team-gpt-4o"],
    )

    try:
        response = client.get(
            "/v1/files",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("api_key") == "team-byok-openai-key"
    assert captured_kwargs.get("custom_llm_provider") == "openai"
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_list_files_with_all_proxy_models_team_uses_openai_deployment(
    mocker: MockerFixture, monkeypatch
):
    """
    Teams with all-proxy-models (or empty models) must still resolve openai
    credentials for plain GET /v1/files.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles, SpecialModelNames

    wildcard_router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "team-openai-key",
                },
            },
            {
                "model_name": "claude-opus-4-6",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-6",
                    "api_key": "anthropic-key",
                },
            },
        ]
    )

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, wildcard_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", wildcard_router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=[])
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_list(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(litellm, "afile_list", _mock_afile_list)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
        team_id="test-team",
        team_models=[SpecialModelNames.all_proxy_models.value],
    )

    try:
        response = client.get(
            "/v1/files",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("api_key") == "team-openai-key"
    assert captured_kwargs.get("custom_llm_provider") == "openai"
    proxy_logging_obj.post_call_failure_hook.assert_not_called()
