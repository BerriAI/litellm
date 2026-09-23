from typing import Final

import pytest
import respx

import litellm

API_BASE: Final = "https://api.x.ai"
KEY: Final = "xai-test-key"


@pytest.fixture(autouse=True)
def _httpx_transport_so_respx_can_intercept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


_XAI_FILE: Final = {
    "bytes": 337,
    "created_at": 1790197740,
    "expires_at": None,
    "filename": "batch.jsonl",
    "id": "file_07",
    "object": "file",
    "purpose": "",
}


@pytest.mark.parametrize("sync_mode", [True, False])
@respx.mock
async def test_create_file_uploads_multipart_to_xai_and_reports_batch_purpose(sync_mode: bool) -> None:
    route: Final = respx.post(f"{API_BASE}/v1/files").respond(200, json=_XAI_FILE)

    kwargs: Final = {
        "file": ("batch.jsonl", b'{"custom_id":"r1"}\n', "application/jsonl"),
        "purpose": "batch",
        "custom_llm_provider": "xai",
        "api_key": KEY,
        "api_base": API_BASE,
    }
    created: Final = litellm.create_file(**kwargs) if sync_mode else await litellm.acreate_file(**kwargs)

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'filename="batch.jsonl"' in request.content
    assert b'{"custom_id":"r1"}' in request.content
    assert created.model_dump(exclude_none=True) == {
        "id": "file_07",
        "bytes": 337,
        "created_at": 1790197740,
        "filename": "batch.jsonl",
        "object": "file",
        "purpose": "batch",
        "status": "uploaded",
    }


@respx.mock
async def test_file_content_of_an_uploaded_file_downloads_original_bytes() -> None:
    respx.get(f"{API_BASE}/v1/files/file_07/content").respond(200, content=b'{"custom_id":"r1"}\n')

    content: Final = await litellm.afile_content(
        file_id="file_07", custom_llm_provider="xai", api_key=KEY, api_base=API_BASE
    )

    assert content.content == b'{"custom_id":"r1"}\n'


@respx.mock
async def test_delete_file_maps_xai_deleted_object() -> None:
    respx.delete(f"{API_BASE}/v1/files/file_07").respond(200, json={"id": "file_07", "deleted": True, "object": "file"})

    deleted: Final = await litellm.afile_delete(
        file_id="file_07", custom_llm_provider="xai", api_key=KEY, api_base=API_BASE
    )

    assert deleted.model_dump() == {"id": "file_07", "deleted": True, "object": "file"}


@respx.mock
async def test_list_files_reads_data_array() -> None:
    respx.get(f"{API_BASE}/v1/files").respond(200, json={"data": [_XAI_FILE], "pagination_token": None})

    listed: Final = await litellm.afile_list(custom_llm_provider="xai", api_key=KEY, api_base=API_BASE)

    assert [f.id for f in listed] == ["file_07"]
