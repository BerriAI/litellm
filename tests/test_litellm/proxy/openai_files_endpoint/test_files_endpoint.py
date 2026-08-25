import json
from typing import Final, List, Optional
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture


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
from litellm.types.llms.openai import (
    FileListPage,
    HttpxBinaryResponseContent,
    OpenAIFileObject,
)

client = TestClient(app)
from litellm.caching.caching import DualCache
from litellm.proxy.proxy_server import hash_token
from litellm.proxy.utils import ProxyLogging

VALID_BATCH_LINE = (
    b'{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions",'
    b' "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]}}\n'
)


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
    error = response.json()["error"]
    assert "Invalid purpose: my-bad-purpose" in error["message"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "purpose"


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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

    test_file = ("mydata.jsonl", VALID_BATCH_LINE, "application/jsonl")
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

    test_file = ("mydata.jsonl", VALID_BATCH_LINE, "application/jsonl")
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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

        async def afile_list(
            self,
            purpose,
            litellm_parent_otel_span,
            user_api_key_dict,
            limit=None,
            after=None,
            **data,
        ):
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


def test_list_files_model_routing_does_not_forward_custom_llm_provider_twice(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=[])
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_list(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(litellm, "afile_list", _mock_afile_list)
    monkeypatch.setattr(
        "litellm.proxy.openai_files_endpoints.files_endpoints.handle_model_based_routing",
        lambda **kwargs: (
            True,
            "azure-gpt-4o",
            None,
            {
                "custom_llm_provider": "azure",
                "api_key": "azure-key",
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
            "/v1/files",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs["custom_llm_provider"] == "azure"
    assert captured_kwargs["api_key"] == "azure-key"
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


def test_unscoped_list_files_uses_managed_file_store(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.proxy._types import LitellmUserRoles

    managed_file = OpenAIFileObject(
        id="unified-file-id",
        object="file",
        bytes=100,
        created_at=1700000000,
        filename="output.jsonl",
        purpose="batch_output",
        status="processed",
    )

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    managed_files = mocker.MagicMock(spec=BaseFileEndpoints)
    managed_files.afile_list = mocker.AsyncMock(
        return_value={
            "object": "list",
            "data": [managed_file],
            "first_id": managed_file.id,
            "last_id": managed_file.id,
            "has_more": False,
        }
    )
    proxy_logging_obj.proxy_hook_mapping["managed_files"] = managed_files
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=None)
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    provider_list = mocker.patch.object(litellm, "afile_list", new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )

    try:
        response = client.get(
            "/v1/files",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["id"] == "unified-file-id"
    managed_files.afile_list.assert_awaited_once()
    assert managed_files.afile_list.await_args.kwargs["user_api_key_dict"].user_id == "test-user"
    assert managed_files.afile_list.await_args.kwargs["limit"] is None
    assert managed_files.afile_list.await_args.kwargs["after"] is None
    provider_list.assert_not_awaited()
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_unscoped_list_files_forwards_limit_and_after_to_the_managed_file_store(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.proxy._types import LitellmUserRoles

    second_page_file = OpenAIFileObject(
        id="unified-file-id-2",
        object="file",
        bytes=100,
        created_at=1700000000,
        filename="output.jsonl",
        purpose="batch",
        status="processed",
    )

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    managed_files = mocker.MagicMock(spec=BaseFileEndpoints)
    managed_files.afile_list = mocker.AsyncMock(
        return_value={
            "object": "list",
            "data": [second_page_file],
            "first_id": second_page_file.id,
            "last_id": second_page_file.id,
            "has_more": True,
        }
    )
    proxy_logging_obj.proxy_hook_mapping["managed_files"] = managed_files
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=None)
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    provider_list = mocker.patch.object(litellm, "afile_list", new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )

    try:
        response = client.get(
            "/v1/files?limit=2&after=unified-file-id-1&purpose=batch",
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["id"] == "unified-file-id-2"
    assert response.json()["has_more"] is True
    call_kwargs = managed_files.afile_list.await_args.kwargs
    assert call_kwargs["limit"] == 2
    assert call_kwargs["after"] == "unified-file-id-1"
    assert call_kwargs["purpose"] == "batch"
    provider_list.assert_not_awaited()
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router: Router, afile_list):
    """Wire GET /v1/files to the managed file store, with afile_list as the store."""
    import litellm.proxy.proxy_server as ps
    from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
    from litellm.proxy._types import LitellmUserRoles

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    managed_files = mocker.MagicMock(spec=BaseFileEndpoints)
    managed_files.afile_list = mocker.AsyncMock(side_effect=afile_list)
    proxy_logging_obj.proxy_hook_mapping["managed_files"] = managed_files
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=None)
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    mocker.patch.object(litellm, "afile_list", new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )
    return managed_files


def _get_list_files(path: str):
    try:
        return client.get(path, headers={"Authorization": "Bearer test-key"})
    finally:
        import litellm.proxy.proxy_server as ps

        app.dependency_overrides.pop(ps.user_api_key_auth, None)


def _get_unscoped_list_files(query: str):
    return _get_list_files(f"/v1/files{query}")


_EMPTY_FILE_LIST_PAGE: Final = {
    "object": "list",
    "data": [],
    "first_id": None,
    "last_id": None,
    "has_more": False,
}


async def _validating_afile_list(**kwargs):
    """Stand in for the managed file store, applying the real request validation."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        validate_file_list_limit,
        validate_file_list_purpose,
    )

    validate_file_list_limit(kwargs.get("limit"))
    validate_file_list_purpose(kwargs.get("purpose"))
    return FileListPage(**_EMPTY_FILE_LIST_PAGE)


async def _permissive_afile_list(**kwargs):
    """Stand in for a file store that validates nothing, so only the route can reject."""
    return FileListPage(**_EMPTY_FILE_LIST_PAGE)


@pytest.mark.parametrize(
    "limit, bound, expected_range",
    [
        (0, "below minimum", ">= 1"),
        (-1, "below minimum", ">= 1"),
        (10001, "above maximum", "<= 10000"),
    ],
)
def test_unscoped_list_files_returns_400_for_a_limit_outside_the_openai_range(
    mocker: MockerFixture, monkeypatch, llm_router: Router, limit, bound, expected_range
):
    """An out-of-range limit is the caller's mistake, so it must not read as a 500 the SDK retries."""
    _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _validating_afile_list)

    response = _get_unscoped_list_files(f"?limit={limit}")

    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {
            "message": (
                f"Invalid 'limit': integer {bound} value. "
                f"Expected a value {expected_range}, but got {limit} instead."
            ),
            "type": "invalid_request_error",
            "param": "limit",
            "code": "400",
        }
    }


@pytest.mark.parametrize("limit", [1, 10000])
def test_unscoped_list_files_accepts_the_ends_of_the_openai_limit_range(
    mocker: MockerFixture, monkeypatch, llm_router: Router, limit
):
    managed_files = _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _validating_afile_list)

    response = _get_unscoped_list_files(f"?limit={limit}")

    assert response.status_code == 200, response.text
    assert response.json()["data"] == []
    assert managed_files.afile_list.await_args.kwargs["limit"] == limit


@pytest.mark.parametrize(
    "path",
    [
        "/v1/files?limit=0",
        "/v1/files?limit=0&target_model_names=gpt-3.5-turbo",
        "/openai/v1/files?limit=0",
    ],
    ids=["managed-file-store", "target-model-names", "provider-route"],
)
def test_list_files_validates_the_limit_on_every_branch(
    mocker: MockerFixture, monkeypatch, llm_router: Router, path
):
    """The limit is a route-level contract, so the scoped and provider branches reject it too."""
    _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _permissive_afile_list)

    response = _get_list_files(path)

    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "limit"
    assert response.json()["error"]["message"] == (
        "Invalid 'limit': integer below minimum value. Expected a value >= 1, but got 0 instead."
    )


def test_unscoped_list_files_returns_400_for_an_unknown_after_cursor(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    from litellm.proxy._types import ProxyException

    async def _unknown_cursor(**kwargs):
        raise ProxyException(
            message=f"Invalid 'after' cursor: no file found with id '{kwargs['after']}'.",
            type="invalid_request_error",
            param="after",
            code=400,
            openai_code="invalid_value",
        )

    _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _unknown_cursor)

    response = _get_unscoped_list_files("?after=file-does-not-exist-xyz")

    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {
            "message": "Invalid 'after' cursor: no file found with id 'file-does-not-exist-xyz'.",
            "type": "invalid_request_error",
            "param": "after",
            "code": "400",
        }
    }


def _managed_file(file_id: str) -> OpenAIFileObject:
    return OpenAIFileObject(
        id=file_id,
        bytes=17,
        created_at=1700000000,
        filename="batch_input.jsonl",
        object="file",
        purpose="batch",
        status="uploaded",
    )


def test_unscoped_list_files_hands_post_call_hooks_a_page_object(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Logging callbacks read ``response.data`` off a listing, so the managed
    branch has to hand them the same page shape the provider branch does. A bare
    mapping turns every registered callback into a 500 on this route."""
    import litellm.proxy.proxy_server as ps

    seen_by_callback: list[list[str]] = []

    async def _reads_response_data(data, user_api_key_dict, response):
        seen_by_callback.append([file.id for file in response.data])
        return None

    async def _one_managed_file(**kwargs):
        return FileListPage(
            data=[_managed_file("unified-file-id")],
            first_id="unified-file-id",
            last_id="unified-file-id",
        )

    _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _one_managed_file)
    ps.proxy_logging_obj.post_call_success_hook = _reads_response_data

    response = _get_unscoped_list_files("")

    assert response.status_code == 200, response.text
    assert seen_by_callback == [["unified-file-id"]]
    body = response.json()
    assert list(body) == ["object", "data", "first_id", "last_id", "has_more"]
    assert body["object"] == "list"
    assert [file["id"] for file in body["data"]] == ["unified-file-id"]
    assert body["has_more"] is False


@pytest.mark.parametrize("purpose", ["nonexistent_purpose", "EVALS", "batch "])
def test_unscoped_list_files_returns_400_for_a_purpose_the_api_never_accepts(
    mocker: MockerFixture, monkeypatch, llm_router: Router, purpose
):
    """An unknown purpose matches nothing, so reporting an empty page would dress
    a bad request up as a successful one. The provider-backed branches reject the
    same values, and so does the upload route."""
    from urllib.parse import quote

    _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _validating_afile_list)

    response = _get_list_files(f"/v1/files?purpose={quote(purpose)}")

    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "purpose"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["message"].startswith(f"Invalid purpose: {purpose}. Must be one of: ")


@pytest.mark.parametrize("purpose", ["batch", "assistants", "fine-tune"])
def test_unscoped_list_files_accepts_every_documented_purpose(
    mocker: MockerFixture, monkeypatch, llm_router: Router, purpose
):
    managed_files = _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _validating_afile_list)

    response = _get_list_files(f"/v1/files?purpose={purpose}")

    assert response.status_code == 200, response.text
    assert managed_files.afile_list.await_args.kwargs["purpose"] == purpose


def test_list_files_reports_a_bad_target_model_names_as_a_400(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """The exception tail reports an HTTPException with its own status and error
    type rather than relabelling it, so a client that branches on either keeps
    reading the same thing off a bad request."""
    _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _permissive_afile_list)

    response = _get_list_files("/v1/files?target_model_names=gpt-3.5-turbo,gpt-4o")

    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {
            "message": "target_model_names on list files must be a list of one model name. Example: ['gpt-4o']",
            "type": "None",
            "param": "None",
            "code": "400",
        }
    }


def test_list_files_reports_an_unexpected_file_store_error_as_a_500(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    async def _blows_up(**kwargs):
        raise RuntimeError("managed file table is unreachable")

    _setup_unscoped_list_files_route(mocker, monkeypatch, llm_router, _blows_up)

    response = _get_unscoped_list_files("")

    assert response.status_code == 500, response.text
    assert response.json()["error"]["message"] == "managed file table is unreachable"


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


def _setup_vertex_named_credential_router(monkeypatch) -> Router:
    from litellm.types.utils import CredentialItem

    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="vertex-named-cred",
                credential_info={},
                credential_values={
                    "vertex_project": "customer-project",
                    "vertex_location": "us-central1",
                    "vertex_credentials": "/creds/customer-sa.json",
                },
            )
        ],
    )
    return Router(
        model_list=[
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "litellm_credential_name": "vertex-named-cred",
                },
            }
        ]
    )


def _assert_vertex_named_credentials_attached(captured_kwargs: dict) -> None:
    assert captured_kwargs.get("custom_llm_provider") == "vertex_ai"
    assert captured_kwargs.get("vertex_project") == "customer-project"
    assert captured_kwargs.get("vertex_location") == "us-central1"
    assert captured_kwargs.get("vertex_credentials") == "/creds/customer-sa.json"
    assert captured_kwargs.get("model") is None


def test_create_file_provider_only_resolves_named_vertex_credentials(
    mocker: MockerFixture, monkeypatch
):
    """
    POST /v1/files with only a custom-llm-provider header (no model, no
    target_model_names) must attach the configured named vertex credential to
    the upstream call instead of falling through to google.auth.default(),
    which uploads into the hosting environment's GCP project.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    router = _setup_vertex_named_credential_router(monkeypatch)
    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_acreate_file(**kwargs):
        captured_kwargs.update(kwargs)
        return OpenAIFileObject(
            id="file-vertex-123",
            object="file",
            bytes=2,
            created_at=1234567890,
            filename="batch.jsonl",
            purpose="batch",
            status="uploaded",
        )

    monkeypatch.setattr(litellm, "acreate_file", _mock_acreate_file)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", VALID_BATCH_LINE, "application/jsonl")},
            data={"purpose": "batch"},
            headers={
                "Authorization": "Bearer test-key",
                "custom-llm-provider": "vertex_ai",
            },
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    _assert_vertex_named_credentials_attached(captured_kwargs)
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_get_file_provider_only_resolves_named_vertex_credentials(
    mocker: MockerFixture, monkeypatch
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    router = _setup_vertex_named_credential_router(monkeypatch)
    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_retrieve(**kwargs):
        captured_kwargs.update(kwargs)
        return OpenAIFileObject(
            id="file-abc123",
            object="file",
            bytes=2,
            created_at=1234567890,
            filename="batch.jsonl",
            purpose="batch",
            status="uploaded",
        )

    monkeypatch.setattr(litellm, "afile_retrieve", _mock_afile_retrieve)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = client.get(
            "/v1/files/file-abc123",
            headers={
                "Authorization": "Bearer test-key",
                "custom-llm-provider": "vertex_ai",
            },
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("file_id") == "file-abc123"
    _assert_vertex_named_credentials_attached(captured_kwargs)
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_get_file_content_provider_only_resolves_named_vertex_credentials(
    mocker: MockerFixture, monkeypatch
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    router = _setup_vertex_named_credential_router(monkeypatch)
    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_content(**kwargs):
        captured_kwargs.update(kwargs)
        return HttpxBinaryResponseContent(
            response=httpx.Response(
                status_code=200,
                content=b"vertex-bytes",
                headers={"content-type": "application/octet-stream"},
            )
        )

    monkeypatch.setattr(litellm, "afile_content", _mock_afile_content)

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
                "custom-llm-provider": "vertex_ai",
            },
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert response.content == b"vertex-bytes"
    assert captured_kwargs.get("file_id") == "file-abc123"
    _assert_vertex_named_credentials_attached(captured_kwargs)
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_delete_file_provider_only_resolves_named_vertex_credentials(
    mocker: MockerFixture, monkeypatch
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    router = _setup_vertex_named_credential_router(monkeypatch)
    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_afile_delete(**kwargs):
        captured_kwargs.update(kwargs)
        return OpenAIFileObject(
            id="file-abc123",
            object="file",
            bytes=2,
            created_at=1234567890,
            filename="batch.jsonl",
            purpose="batch",
            status="uploaded",
        )

    monkeypatch.setattr(litellm, "afile_delete", _mock_afile_delete)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )

    try:
        response = client.delete(
            "/v1/files/file-abc123",
            headers={
                "Authorization": "Bearer test-key",
                "custom-llm-provider": "vertex_ai",
            },
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("file_id") == "file-abc123"
    _assert_vertex_named_credentials_attached(captured_kwargs)
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_create_file_provider_only_skips_other_team_vertex_deployment(
    mocker: MockerFixture, monkeypatch
):
    """
    Regression: with a team-scoped vertex deployment indexed before a global
    one under the same model name, a provider-only upload from a different
    team must use the global deployment's credentials, never the other
    team's.
    """
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    router = Router(
        model_list=[
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "vertex_project": "team-b-project",
                },
                "model_info": {
                    "id": "team-b-vertex",
                    "team_id": "team-b",
                    "team_public_model_name": "gemini-2.5-pro",
                },
            },
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "vertex_project": "shared-project",
                },
            },
        ]
    )
    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()

    captured_kwargs: dict = {}

    async def _mock_acreate_file(**kwargs):
        captured_kwargs.update(kwargs)
        return OpenAIFileObject(
            id="file-vertex-456",
            object="file",
            bytes=2,
            created_at=1234567890,
            filename="batch.jsonl",
            purpose="batch",
            status="uploaded",
        )

    monkeypatch.setattr(litellm, "acreate_file", _mock_acreate_file)

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
        team_id="team-a",
        team_models=["gemini-2.5-pro"],
    )

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", VALID_BATCH_LINE, "application/jsonl")},
            data={"purpose": "batch"},
            headers={
                "Authorization": "Bearer test-key",
                "custom-llm-provider": "vertex_ai",
            },
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert captured_kwargs.get("vertex_project") == "shared-project"
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def _team_openai_plus_global_anthropic_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "team-gpt",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "team-openai-key",
                },
                "model_info": {
                    "id": "team-a-openai",
                    "team_id": "team-a",
                    "team_public_model_name": "team-gpt",
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


def _list_files_captured_kwargs(
    mocker: MockerFixture, monkeypatch, router: Router, key_models: list
) -> dict:
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

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
        team_id="team-a",
        team_models=["team-gpt", "claude-opus-4-6"],
        models=key_models,
    )

    try:
        response = client.get(
            "/v1/files",
            headers={
                "Authorization": "Bearer test-key",
                "custom-llm-provider": "openai",
            },
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    return captured_kwargs


def test_list_files_key_restricted_to_other_provider_does_not_leak_team_openai_credentials(
    mocker: MockerFixture, monkeypatch
):
    """
    Regression: a key restricted to an anthropic model on a team that also has
    an openai deployment must not attach the team's openai credentials to a
    provider-only openai files call; key-level model restrictions apply to
    credential resolution, not just completions.
    """
    captured_kwargs = _list_files_captured_kwargs(
        mocker, monkeypatch, _team_openai_plus_global_anthropic_router(), ["claude-opus-4-6"]
    )
    assert captured_kwargs.get("api_key") != "team-openai-key"


def test_list_files_key_allowed_openai_model_still_resolves_team_credentials(
    mocker: MockerFixture, monkeypatch
):
    """
    A key whose allowlist includes the team's openai model keeps resolving that
    deployment's credentials for provider-only openai files calls.
    """
    captured_kwargs = _list_files_captured_kwargs(
        mocker, monkeypatch, _team_openai_plus_global_anthropic_router(), ["team-gpt"]
    )
    assert captured_kwargs.get("api_key") == "team-openai-key"


@pytest.mark.parametrize(
    "http_method, url, patched_litellm_call",
    [
        ("get", "/v1/files/file-victim-abc123", "litellm.afile_retrieve"),
        ("get", "/v1/files/file-victim-abc123/content", "litellm.afile_content"),
        ("delete", "/v1/files/file-victim-abc123", "litellm.afile_delete"),
    ],
)
def test_require_managed_files_rejects_raw_provider_file_id(
    mocker: MockerFixture,
    monkeypatch,
    llm_router: Router,
    http_method: str,
    url: str,
    patched_litellm_call: str,
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.require_managed_files", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    mock_call = mocker.patch(patched_litellm_call, new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="attacker-user"
    )

    try:
        response = getattr(client, http_method)(
            url, headers={"Authorization": "Bearer test-key"}
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        monkeypatch.setattr("litellm.require_managed_files", False)

    assert response.status_code == 400, response.text
    mock_call.assert_not_called()


def test_get_file_content_model_routed_attaches_trusted_model_credentials(monkeypatch):
    """A managed batch output id routes by model, and that branch must build the snapshot.

    The managed-files pre-call hook sets data["model"] for any id carrying
    llm_output_file_id, so batch output retrieval always takes the model-routed branch
    and never reaches managed_files_obj.afile_content. Bedrock resolves its output
    bucket only from _litellm_internal_model_credentials, so without the snapshot every
    Bedrock batch output retrieval fails with "S3 bucket_name is required".
    """
    import base64
    from types import MappingProxyType

    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles
    from litellm.types.utils import SpecialEnums

    router = Router(
        model_list=[
            {
                "model_name": "anthropic.batch.claude-4.5-haiku",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
                    "aws_region_name": "us-east-1",
                    "s3_bucket_name": "configured-batch-bucket",
                },
                "model_info": {"id": "bedrock-batch-deployment-id"},
            }
        ]
    )

    from unittest.mock import MagicMock

    managed_file_row = MagicMock()
    managed_file_row.created_by = "test-user"
    managed_file_row.team_id = None
    managed_file_row.storage_backend = None
    managed_file_row.storage_url = None
    prisma_stub = MagicMock()
    prisma_stub.db.litellm_managedfiletable.find_first = AsyncMock(return_value=managed_file_row)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_stub)
    setup_proxy_logging_object(monkeypatch, router)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)

    # One frozen snapshot per call rather than one dict merged across calls, so a second
    # invocation is visible instead of silently overwriting the first.
    calls: list[MappingProxyType] = []

    async def _mock_router_afile_content(**kwargs):
        calls.append(MappingProxyType(dict(kwargs)))
        return HttpxBinaryResponseContent(
            response=httpx.Response(
                status_code=200,
                content=b'{"recordId":"req-1"}',
                headers={"content-type": "application/octet-stream"},
            )
        )

    monkeypatch.setattr(router, "afile_content", _mock_router_afile_content)

    unified_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/jsonl",
        "unified-output-id",
        "anthropic.batch.claude-4.5-haiku",
        "llm_output_file_id,s3://configured-batch-bucket/out/batch.jsonl",
        "bedrock-batch-deployment-id",
    )
    encoded_id = base64.urlsafe_b64encode(unified_id.encode()).decode().rstrip("=")

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="test-user",
    )
    try:
        response = client.get(
            f"/v1/files/{encoded_id}/content",
            headers={"Authorization": "Bearer test-key", "custom-llm-provider": "bedrock"},
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    assert len(calls) == 1, f"expected exactly one routed retrieval, got {len(calls)}"
    snapshot = calls[0].get("_litellm_internal_model_credentials")
    assert snapshot is not None, "model-routed branch must attach the trusted credential snapshot"
    assert isinstance(
        snapshot, MappingProxyType
    ), "snapshot must be a MappingProxyType; a plain dict is rejected by get_configured_s3_bucket_name"
    assert snapshot["s3_bucket_name"] == "configured-batch-bucket"


def _unified_managed_file_id() -> str:
    import base64

    from litellm.types.utils import SpecialEnums

    unified_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json", "victim-unified-id", "gpt-3.5-turbo", "file-victim-abc123", "gpt-3.5-turbo-id"
    )
    return base64.urlsafe_b64encode(unified_id.encode()).decode().rstrip("=")


class _ManagedResourceAccessCheckerStub:
    async def can_user_call_unified_file_id(
        self,
        unified_file_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        return True

    async def can_user_call_unified_object_id(
        self,
        unified_object_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> bool:
        return True


@pytest.mark.asyncio
async def test_require_managed_files_allows_owned_unified_managed_file_id(monkeypatch):
    from litellm.proxy.openai_files_endpoints.common_utils import (
        validate_managed_id_requirement,
    )

    monkeypatch.setattr("litellm.require_managed_files", True)

    await validate_managed_id_requirement(
        resource_id=_unified_managed_file_id(),
        resource_kind="file",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="owner-user"),
        managed_files_obj=_ManagedResourceAccessCheckerStub(),
    )


@pytest.mark.asyncio
async def test_managed_file_id_requirement_is_opt_in(monkeypatch):
    from litellm.proxy.openai_files_endpoints.common_utils import (
        validate_managed_id_requirement,
    )

    monkeypatch.setattr("litellm.require_managed_files", False)

    await validate_managed_id_requirement(
        resource_id="file-victim-abc123",
        resource_kind="file",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        managed_files_obj=None,
    )


def test_raw_provider_file_id_retrieve_allowed_when_managed_files_not_required(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles

    monkeypatch.setattr("litellm.require_managed_files", False)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    mock_retrieve = mocker.patch(
        "litellm.afile_retrieve",
        new=mocker.AsyncMock(
            return_value=OpenAIFileObject(
                id="file-victim-abc123",
                object="file",
                bytes=3,
                created_at=1234567890,
                filename="test.txt",
                purpose="user_data",
                status="uploaded",
            )
        ),
    )

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="some-user"
    )

    try:
        response = client.get(
            "/v1/files/file-victim-abc123", headers={"Authorization": "Bearer test-key"}
        )
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)

    assert response.status_code == 200, response.text
    mock_retrieve.assert_called_once()


def _setup_batch_upload_endpoint(monkeypatch, llm_router: Router) -> list:
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.openai_files_endpoints import files_endpoints as fe

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)

    forwarded_calls: list = []

    async def fake_route_create_file(**kwargs):
        forwarded_calls.append(kwargs)
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
    return forwarded_calls


def _teardown_batch_upload_endpoint():
    import litellm.proxy.proxy_server as ps

    app.dependency_overrides.pop(ps.user_api_key_auth, None)


def test_create_file_batch_over_max_batch_file_size_mb_rejected_before_forwarding(
    monkeypatch, llm_router: Router
):
    import litellm.proxy.proxy_server as ps

    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)
    monkeypatch.setitem(ps.general_settings, "max_batch_file_size_mb", 1)

    oversized = VALID_BATCH_LINE * (2 * 1024 * 1024 // len(VALID_BATCH_LINE) + 1)
    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", oversized, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()

    assert response.status_code == 413, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "file"
    assert "max_batch_file_size_mb" in error["message"]
    assert "1 MB" in error["message"]
    assert forwarded_calls == []


def test_create_file_batch_under_max_batch_file_size_mb_forwards(monkeypatch, llm_router: Router):
    import litellm.proxy.proxy_server as ps

    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)
    monkeypatch.setitem(ps.general_settings, "max_batch_file_size_mb", 1)

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", VALID_BATCH_LINE, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()

    assert response.status_code == 200, response.text
    assert len(forwarded_calls) == 1


def test_create_file_batch_wrong_extension_rejected_before_forwarding(monkeypatch, llm_router: Router):
    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.csv", VALID_BATCH_LINE, "text/csv")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "file"
    assert "batch.csv" in error["message"]
    assert ".jsonl" in error["message"]
    assert forwarded_calls == []


def test_create_file_batch_missing_line_key_rejected_before_forwarding(monkeypatch, llm_router: Router):
    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)

    bad_line = b'{"custom_id": "req-1", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo"}}\n'
    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", VALID_BATCH_LINE + bad_line, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "method"
    assert "line 2" in error["message"]
    assert forwarded_calls == []


def test_create_file_batch_invalid_json_line_rejected_before_forwarding(monkeypatch, llm_router: Router):
    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", b"this is not jsonl\n", "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "file"
    assert "line 1" in error["message"]
    assert "not valid JSON" in error["message"]
    assert forwarded_calls == []


def test_create_file_non_batch_purpose_skips_batch_validation(monkeypatch, llm_router: Router):
    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)

    try:
        response = client.post(
            "/v1/files",
            files={"file": ("notes.txt", b"plain text, not jsonl", "text/plain")},
            data={"purpose": "user_data"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()

    assert response.status_code == 200, response.text
    assert len(forwarded_calls) == 1


def _batch_upload(client_, content: bytes, purpose: str = "batch"):
    return client_.post(
        "/v1/files",
        files={"file": ("batch.jsonl", content, "application/jsonl")},
        data={"purpose": purpose},
        headers={"Authorization": "Bearer test-key"},
    )


@pytest.mark.parametrize(
    "content, purpose, expected_status, expected_fragment",
    [
        (
            b'{"custom_id":"r-0","method":"POST","url":"/v1/chat/completions",'
            b'"body":{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"hi"}]}}\n',
            "batch",
            200,
            None,
        ),
        (
            b'{"custom_id":"r-0","method":"POST","url":"/v1/chat/completions",'
            b'"body":{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"leak me"}]}}\n',
            "batch",
            200,
            None,
        ),
        (
            b'{"custom_id":"r-0","method":"POST","url":"/v1/chat/completions",'
            b'"body":{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"leak me"}]}}\n',
            "assistants",
            200,
            None,
        ),
        (b"{ not json\n", "batch", 400, "line 1"),
    ],
)
def test_batch_upload_runs_guardrails_on_each_record(
    monkeypatch, llm_router: Router, content, purpose, expected_status, expected_fragment
):
    """POST /v1/files with purpose=batch must reach the guardrail chain; other purposes must not."""
    import litellm
    import litellm.proxy.openai_files_endpoints.files_endpoints as fe
    import litellm.proxy.proxy_server as ps
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.utils import ProxyLogging

    class _Redactor(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            for message in data.get("messages") or []:
                if isinstance(message.get("content"), str) and "leak" in message["content"]:
                    message["content"] = "***"
            return data

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)
    monkeypatch.setattr(litellm, "callbacks", [_Redactor(guardrail_name="g", default_on=True)])
    ProxyLogging._callback_capabilities_cache.clear()

    async def fake_route_create_file(**kwargs):
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
    try:
        resp = _batch_upload(client, content, purpose)
        assert resp.status_code == expected_status, resp.text
        if expected_fragment is not None:
            assert expected_fragment in resp.text
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        ProxyLogging._callback_capabilities_cache.clear()


def test_batch_upload_redacts_per_record(monkeypatch, llm_router: Router):
    """An offending record is submitted masked, matching what the online path does per request."""
    expected_custom_ids = ["keep-1", "dirty", "keep-2"]
    import json as _json

    import litellm
    import litellm.proxy.openai_files_endpoints.files_endpoints as fe
    import litellm.proxy.proxy_server as ps
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.utils import ProxyLogging

    class _Redactor(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            for message in data.get("messages") or []:
                if isinstance(message.get("content"), str) and "leak" in message["content"]:
                    message["content"] = message["content"].replace("leak", "***")
            return data

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)
    monkeypatch.setattr(litellm, "callbacks", [_Redactor(guardrail_name="g", default_on=True)])
    ProxyLogging._callback_capabilities_cache.clear()

    uploaded = {}

    async def fake_route_create_file(**kwargs):
        handle = kwargs["_create_file_request"]["file"][1]
        uploaded["body"] = handle.read() if hasattr(handle, "read") else handle
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

    def _row(custom_id, content):
        return _json.dumps(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": content}]},
            }
        )

    content = ("\n".join([_row("keep-1", "fine"), _row("dirty", "please leak this"), _row("keep-2", "fine")])).encode()
    try:
        resp = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", content, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 200, resp.text
        rows = [_json.loads(line) for line in uploaded["body"].decode().splitlines()]
        assert [row["custom_id"] for row in rows] == expected_custom_ids
        assert rows[1]["body"]["messages"][0]["content"] == "please *** this"
        report = resp.json()["litellm_batch_guardrail"]
        assert report["submitted_records"] == 3
        assert report["modified_records"] == [
            {"line": 2, "custom_id": "dirty", "action": "redacted", "guardrail": None}
        ]
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        ProxyLogging._callback_capabilities_cache.clear()


PLAIN_UPLOAD_RESPONSE_BODY = {
    "id": "dummy-id",
    "object": "file",
    "bytes": 0,
    "created_at": 1234567890,
    "filename": "batch.jsonl",
    "purpose": "batch",
    "status": "uploaded",
    "expires_at": None,
    "status_details": None,
}


def test_create_file_omits_batch_guardrail_field_when_no_guardrail_configured(monkeypatch, llm_router: Router):
    """An upload no guardrail is configured for serialises the plain OpenAI file shape."""
    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)
    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", VALID_BATCH_LINE, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()

    assert response.status_code == 200, response.text
    assert len(forwarded_calls) == 1
    assert response.json() == PLAIN_UPLOAD_RESPONSE_BODY


def test_create_file_omits_batch_guardrail_field_when_guardrail_made_no_changes(monkeypatch, llm_router: Router):
    """A guardrail that runs and changes nothing leaves the response the plain OpenAI file shape."""
    import litellm
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy.utils import ProxyLogging

    class _Passthrough(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            return data

    forwarded_calls = _setup_batch_upload_endpoint(monkeypatch, llm_router)
    monkeypatch.setattr(litellm, "callbacks", [_Passthrough(guardrail_name="noop", default_on=True)])
    ProxyLogging._callback_capabilities_cache.clear()
    try:
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", VALID_BATCH_LINE, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
    finally:
        _teardown_batch_upload_endpoint()
        ProxyLogging._callback_capabilities_cache.clear()

    assert response.status_code == 200, response.text
    assert len(forwarded_calls) == 1
    assert response.json() == PLAIN_UPLOAD_RESPONSE_BODY


def test_batch_upload_closes_the_spools_it_opened(monkeypatch, llm_router: Router):
    """The scan and the rewrite each open a spool; the request owns both and must not leak them."""
    import json as _json

    import litellm
    import litellm.proxy.openai_files_endpoints.batch_guardrails as bg
    import litellm.proxy.openai_files_endpoints.files_endpoints as fe
    import litellm.proxy.proxy_server as ps
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.utils import ProxyLogging

    class _Redactor(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            for message in data.get("messages") or []:
                if "leak" in (message.get("content") or ""):
                    message["content"] = message["content"].replace("leak", "***")
            return data

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    setup_proxy_logging_object(monkeypatch, llm_router)
    monkeypatch.setattr(litellm, "callbacks", [_Redactor(guardrail_name="g", default_on=True)])
    ProxyLogging._callback_capabilities_cache.clear()

    spools = []
    real = bg.tempfile.SpooledTemporaryFile

    def _tracking(*args, **kwargs):
        handle = real(*args, **kwargs)
        spools.append(handle)
        return handle

    monkeypatch.setattr(bg.tempfile, "SpooledTemporaryFile", _tracking)

    async def fake_route_create_file(**kwargs):
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

    def _row(custom_id, content):
        return _json.dumps(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": content}]},
            }
        )

    content = ("\n".join([_row("keep", "fine"), _row("dirty", "please leak this")])).encode()
    try:
        resp = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", content, "application/jsonl")},
            data={"purpose": "batch"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 200, resp.text
        assert len(spools) == 2, f"expected a scan spool and a rewrite spool, saw {len(spools)}"
        assert all(handle.closed for handle in spools), "the request must close every spool it opened"
    finally:
        app.dependency_overrides.pop(ps.user_api_key_auth, None)
        ProxyLogging._callback_capabilities_cache.clear()


def _managed_file_row(
    unified_file_id: str,
    stored_file_id: Optional[str] = None,
    created_by: str = "test-user",
    team_id: Optional[str] = None,
) -> MagicMock:
    file_object = OpenAIFileObject(
        id=stored_file_id or unified_file_id,
        bytes=100,
        created_at=1700000000,
        filename="batch.jsonl",
        object="file",
        purpose="batch",
        status="processed",
    )
    return MagicMock(
        unified_file_id=unified_file_id,
        file_object=file_object.model_dump(),
        created_by=created_by,
        team_id=team_id,
    )


def _row_matches_where(row, where) -> bool:
    for field, expected in where.items():
        if field == "OR":
            if not any(_row_matches_where(row, clause) for clause in expected):
                return False
        elif getattr(row, field) != expected:
            return False
    return True


class _ManagedFileTableOverRows:
    """Enough of the Prisma table for the real hook's owner-scoped keyset query."""

    def __init__(self, rows):
        self.rows = list(rows)

    def _owned_rows(self, where):
        return [row for row in self.rows if _row_matches_where(row, where)]

    async def find_first(self, where):
        return next(iter(self._owned_rows(where)), None)

    async def find_many(self, where, take=None, order=None, cursor=None, skip=0):
        rows = self._owned_rows(where)
        if cursor is not None:
            start = next(
                index
                for index, row in enumerate(rows)
                if row.unified_file_id == cursor["unified_file_id"]
            )
            rows = rows[start + skip :]
        return rows if take is None else rows[:take]


def _setup_unscoped_list_files_route_over_real_hook(
    mocker, monkeypatch, llm_router: Router, rows
):
    """Wire GET /v1/files to a real managed-files hook over `rows`, with no provider
    credentials in the process. The neighbouring setup stubs the hook with a
    MagicMock, so it cannot see anything past the route's argument plumbing."""
    import litellm.proxy.proxy_server as ps
    from litellm.proxy._types import LitellmUserRoles
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    for env_var in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORGANIZATION"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(litellm, "api_key", None, raising=False)
    monkeypatch.setattr(litellm, "openai_key", None, raising=False)

    managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(), prisma_client=MagicMock()
    )
    managed_files.prisma_client.db.litellm_managedfiletable = _ManagedFileTableOverRows(rows)

    proxy_logging_obj = setup_proxy_logging_object(monkeypatch, llm_router)
    proxy_logging_obj.proxy_hook_mapping["managed_files"] = managed_files
    proxy_logging_obj.update_request_status = mocker.AsyncMock()
    proxy_logging_obj.post_call_success_hook = mocker.AsyncMock(return_value=None)
    proxy_logging_obj.post_call_failure_hook = mocker.AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)

    provider_list = mocker.patch.object(litellm, "afile_list", new=mocker.AsyncMock())

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="test-key",
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )
    return proxy_logging_obj, provider_list


def test_unscoped_list_files_reads_the_store_without_any_provider_key(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """The exact shape `client.files.list()` produces, with no target_model_names, no
    provider and no OPENAI_API_KEY in the process, must return the caller's managed
    files instead of falling through to a keyless provider client (#35362)."""
    proxy_logging_obj, provider_list = _setup_unscoped_list_files_route_over_real_hook(
        mocker, monkeypatch, llm_router, [_managed_file_row("unified-file-1")]
    )

    response = _get_unscoped_list_files("")

    assert response.status_code == 200, response.text
    assert [file["id"] for file in response.json()["data"]] == ["unified-file-1"]
    provider_list.assert_not_awaited()
    proxy_logging_obj.post_call_failure_hook.assert_not_called()


def test_unscoped_list_files_returns_unified_ids_for_rows_storing_a_raw_provider_id(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Batch output rows store the provider's file object, whose id is a raw `file-`
    the caller cannot act on. The listing hands back the row's unified id so
    files.retrieve and files.content work on what it returned."""
    _setup_unscoped_list_files_route_over_real_hook(
        mocker,
        monkeypatch,
        llm_router,
        [_managed_file_row("unified-batch-output", stored_file_id="file-raw-provider-123")],
    )

    response = _get_unscoped_list_files("")

    assert response.status_code == 200, response.text
    assert [file["id"] for file in response.json()["data"]] == ["unified-batch-output"]
    assert response.json()["data"][0]["filename"] == "batch.jsonl"


def test_unscoped_list_files_does_not_leak_another_callers_files(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """Listing without a model pinned is still owner-scoped."""
    _setup_unscoped_list_files_route_over_real_hook(
        mocker,
        monkeypatch,
        llm_router,
        [
            _managed_file_row("unified-mine"),
            _managed_file_row("unified-theirs", created_by="other-user"),
        ],
    )

    response = _get_unscoped_list_files("")

    assert response.status_code == 200, response.text
    assert [file["id"] for file in response.json()["data"]] == ["unified-mine"]


def test_scoped_list_files_still_resolves_deployment_credentials(
    mocker: MockerFixture, monkeypatch, llm_router: Router
):
    """target_model_names keeps routing to the provider on deployment credentials, so
    the unscoped path does not swallow that route."""
    _, provider_list = _setup_unscoped_list_files_route_over_real_hook(
        mocker, monkeypatch, llm_router, [_managed_file_row("unified-file-1")]
    )
    provider_list.return_value = []

    response = _get_unscoped_list_files("?target_model_names=gpt-3.5-turbo")

    assert response.status_code == 200, response.text
    provider_list.assert_awaited_once()
    assert provider_list.await_args.kwargs["custom_llm_provider"] == "openai"
    assert provider_list.await_args.kwargs["api_key"] == "openai_api_key"
