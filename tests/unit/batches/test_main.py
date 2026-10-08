import datetime
from datetime import timezone
import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

_OPENAI_FILE_JSON: Final = {
    "id": "file-abc123",
    "object": "file",
    "purpose": "batch",
    "filename": "batch.jsonl",
    "bytes": 416,
    "created_at": 1739598666,
    "status": "processed",
}
_OPENAI_BATCH_JSON: Final = {
    "id": "batch_abc123",
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "input_file_id": "file-abc123",
    "status": "validating",
    "completion_window": "24h",
    "created_at": 1739598666,
}


@pytest.mark.asyncio
async def test_acreate_batch_full_crud_and_logging_metadata(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.logging_callback_manager._reset_all_callbacks()

    respx_mock.post("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(200, json=_OPENAI_FILE_JSON)
    )
    respx_mock.post("https://api.openai.com/v1/batches").mock(
        return_value=httpx.Response(200, json=_OPENAI_BATCH_JSON)
    )
    respx_mock.get("https://api.openai.com/v1/batches/batch_abc123").mock(
        return_value=httpx.Response(200, json=_OPENAI_BATCH_JSON)
    )
    respx_mock.get("https://api.openai.com/v1/batches").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [_OPENAI_BATCH_JSON]}
        )
    )
    respx_mock.get("https://api.openai.com/v1/files/file-abc123/content").mock(
        return_value=httpx.Response(200, content=b'{"custom_id": "request-1"}\n')
    )
    respx_mock.get("https://api.openai.com/v1/files/file-abc123").mock(
        return_value=httpx.Response(200, json=_OPENAI_FILE_JSON)
    )
    respx_mock.delete("https://api.openai.com/v1/files/file-abc123").mock(
        return_value=httpx.Response(
            200, json={"id": "file-abc123", "object": "file", "deleted": True}
        )
    )
    respx_mock.get("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [_OPENAI_FILE_JSON]}
        )
    )
    respx_mock.post("https://api.openai.com/v1/batches/batch_abc123/cancel").mock(
        return_value=httpx.Response(
            200, json={**_OPENAI_BATCH_JSON, "status": "cancelling"}
        )
    )

    batch_file: Final = (
        "batch.jsonl",
        b'{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", '
        b'"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}}\n',
        "application/jsonl",
    )
    file_obj: Final = await litellm.acreate_file(
        file=batch_file,
        purpose="batch",
        custom_llm_provider="openai",
        api_key="fake-key",
    )
    batch_input_file_id: Final = file_obj.id
    assert batch_input_file_id is not None

    extra_metadata_field: Final = {
        "user_api_key_alias": "special_api_key_alias",
        "user_api_key_team_alias": "special_team_alias",
    }
    create_batch_response: Final = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        custom_llm_provider="openai",
        api_key="fake-key",
        litellm_metadata={**extra_metadata_field, "key1": "value1", "key2": "value2"},
    )

    assert create_batch_response.id is not None
    assert create_batch_response.endpoint in ("/v1/chat/completions", "/chat/completions")
    assert create_batch_response.input_file_id == batch_input_file_id

    payload: Final = get_logging_payload(
        kwargs={
            "model": "gpt-4o",
            "litellm_params": {
                "metadata": {},
                "litellm_metadata": {
                    **extra_metadata_field,
                    "key1": "value1",
                    "key2": "value2",
                },
            },
        },
        response_obj=create_batch_response,
        start_time=datetime.datetime.now(timezone.utc),
        end_time=datetime.datetime.now(timezone.utc),
    )
    standard_logging_object: Final = {"metadata": json.loads(payload["metadata"])}
    assert (
        standard_logging_object["metadata"]["user_api_key_alias"]
        == extra_metadata_field["user_api_key_alias"]
    )
    assert (
        standard_logging_object["metadata"]["user_api_key_team_alias"]
        == extra_metadata_field["user_api_key_team_alias"]
    )

    retrieved_batch: Final = await litellm.aretrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert retrieved_batch.id == create_batch_response.id

    list_batches: Final = await litellm.alist_batches(
        custom_llm_provider="openai", limit=2, api_key="fake-key"
    )
    assert list_batches is not None

    file_content: Final = await litellm.afile_content(
        file_id=batch_input_file_id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert file_content.content == b'{"custom_id": "request-1"}\n'

    retrieved_file: Final = await litellm.afile_retrieve(
        file_id=batch_input_file_id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert retrieved_file.id == batch_input_file_id

    delete_file_response: Final = await litellm.afile_delete(
        file_id=batch_input_file_id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert delete_file_response.id == batch_input_file_id

    all_files_list: Final = await litellm.afile_list(
        custom_llm_provider="openai", api_key="fake-key"
    )
    assert all_files_list is not None

    cancel_batch_response: Final = await litellm.acancel_batch(
        batch_id=create_batch_response.id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert cancel_batch_response.id == create_batch_response.id


@pytest.mark.asyncio
async def test_delete_batch_output_file(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    batch_with_output: Final = {
        **_OPENAI_BATCH_JSON,
        "status": "completed",
        "output_file_id": "file-output123",
    }
    respx_mock.get("https://api.openai.com/v1/batches/batch_abc123").mock(
        return_value=httpx.Response(200, json=batch_with_output)
    )
    delete_route: Final = respx_mock.delete(
        "https://api.openai.com/v1/files/file-output123"
    ).mock(
        return_value=httpx.Response(
            200, json={"id": "file-output123", "object": "file", "deleted": True}
        )
    )

    batch: Final = await litellm.aretrieve_batch(
        batch_id="batch_abc123", custom_llm_provider="openai", api_key="fake-key"
    )
    assert batch.output_file_id == "file-output123"

    delete_response: Final = await litellm.afile_delete(
        file_id=batch.output_file_id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert delete_route.called
    assert delete_response.id == "file-output123"
