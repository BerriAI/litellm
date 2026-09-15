import pytest

from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints import storage_backend_service
from litellm.proxy.openai_files_endpoints.storage_backend_service import (
    StorageBackendFileService,
)


class _RecordingStorageBackend:
    def __init__(self):
        self.upload_calls = []

    async def upload_file(self, **kwargs):
        self.upload_calls.append(kwargs)
        return "https://storage.example/blob-1"


class _FakeManagedFilesHook(BaseFileEndpoints):
    def __init__(self):
        self.stored = []

    async def acreate_file(
        self, create_file_request, llm_router, target_model_names_list, litellm_parent_otel_span, user_api_key_dict
    ):
        raise NotImplementedError

    async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router=None):
        raise NotImplementedError

    async def afile_list(self, purpose, litellm_parent_otel_span, **data):
        raise NotImplementedError

    async def afile_delete(self, file_id, litellm_parent_otel_span, llm_router, **data):
        raise NotImplementedError

    async def afile_content(self, file_id, litellm_parent_otel_span, llm_router, **data):
        raise NotImplementedError

    async def store_unified_file_id(self, **kwargs):
        self.stored.append(kwargs)


class _FakeProxyLogging:
    def __init__(self, hook):
        self._hook = hook

    def get_proxy_hook(self, hook_name):
        return self._hook if hook_name == "managed_files" else None


def _file_data():
    return {"content": b"x", "filename": "input.jsonl", "content_type": "application/jsonl"}


@pytest.mark.asyncio
async def test_upload_with_target_model_names_but_no_hook_raises_before_uploading(monkeypatch):
    backend = _RecordingStorageBackend()
    monkeypatch.setattr(storage_backend_service, "get_storage_backend", lambda name: backend)

    with pytest.raises(ProxyException) as exc_info:
        await StorageBackendFileService.upload_file_to_storage_backend(
            file_data=_file_data(),
            target_storage="azure_storage",
            target_model_names=["gpt-x"],
            purpose="batch",
            proxy_logging_obj=_FakeProxyLogging(hook=None),
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )

    snapshot = {
        "code": exc_info.value.code,
        "message_names_requirement": "requires a database-connected proxy" in exc_info.value.message,
        "upload_calls": backend.upload_calls,
    }
    assert snapshot == {"code": "400", "message_names_requirement": True, "upload_calls": []}


@pytest.mark.asyncio
async def test_upload_without_target_model_names_skips_hook_requirement(monkeypatch):
    backend = _RecordingStorageBackend()
    monkeypatch.setattr(storage_backend_service, "get_storage_backend", lambda name: backend)

    file_object = await StorageBackendFileService.upload_file_to_storage_backend(
        file_data=_file_data(),
        target_storage="azure_storage",
        target_model_names=[],
        purpose="batch",
        proxy_logging_obj=_FakeProxyLogging(hook=None),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )

    snapshot = {
        "upload_count": len(backend.upload_calls),
        "id_prefix": file_object.id.split("-")[0],
    }
    assert snapshot == {"upload_count": 1, "id_prefix": "file"}


@pytest.mark.asyncio
async def test_upload_with_target_model_names_and_hook_stores_unified_id(monkeypatch):
    backend = _RecordingStorageBackend()
    monkeypatch.setattr(storage_backend_service, "get_storage_backend", lambda name: backend)
    hook = _FakeManagedFilesHook()

    file_object = await StorageBackendFileService.upload_file_to_storage_backend(
        file_data=_file_data(),
        target_storage="azure_storage",
        target_model_names=["gpt-x"],
        purpose="batch",
        proxy_logging_obj=_FakeProxyLogging(hook=hook),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )

    snapshot = {
        "upload_count": len(backend.upload_calls),
        "store_count": len(hook.stored),
        "stored_id_matches_response": hook.stored[0]["file_id"] == file_object.id,
        "model_mappings": hook.stored[0]["model_mappings"],
    }
    assert snapshot == {
        "upload_count": 1,
        "store_count": 1,
        "stored_id_matches_response": True,
        "model_mappings": {"gpt-x": "https://storage.example/blob-1"},
    }
