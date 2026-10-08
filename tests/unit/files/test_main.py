from typing import Final

import httpx
import pytest

import litellm

_FILE_BODY: Final = b'{"prompt": "Hello", "completion": "Hi"}'


@pytest.mark.asyncio
async def test_openai_file_operations_roundtrip(respx_mock):
    files_route: Final = respx_mock.post(url__regex=r".*api\.openai\.com/v1/files$").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "file-abc123",
                "object": "file",
                "bytes": len(_FILE_BODY),
                "created_at": 1699000000,
                "filename": "mydata.jsonl",
                "purpose": "fine-tune",
            },
        )
    )
    list_route: Final = respx_mock.get(url__regex=r".*api\.openai\.com/v1/files$").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "file-abc123",
                        "object": "file",
                        "bytes": len(_FILE_BODY),
                        "created_at": 1699000000,
                        "filename": "mydata.jsonl",
                        "purpose": "fine-tune",
                    }
                ],
            },
        )
    )
    retrieve_route: Final = respx_mock.get(url__regex=r".*api\.openai\.com/v1/files/file-abc123$").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "file-abc123",
                "object": "file",
                "bytes": len(_FILE_BODY),
                "created_at": 1699000000,
                "filename": "mydata.jsonl",
                "purpose": "fine-tune",
            },
        )
    )
    content_route: Final = respx_mock.get(url__regex=r".*api\.openai\.com/v1/files/file-abc123/content.*").mock(
        return_value=httpx.Response(200, content=_FILE_BODY)
    )
    delete_route: Final = respx_mock.delete(url__regex=r".*api\.openai\.com/v1/files/file-abc123.*").mock(
        return_value=httpx.Response(200, json={"id": "file-abc123", "object": "file", "deleted": True})
    )

    uploaded: Final = await litellm.acreate_file(
        file=_FILE_BODY, purpose="fine-tune", custom_llm_provider="openai", api_key="fake-key"
    )
    assert files_route.called
    assert uploaded.id == "file-abc123"

    listed: Final = await litellm.afile_list(custom_llm_provider="openai", api_key="fake-key")
    assert list_route.called
    assert [file.id for file in listed.data] == ["file-abc123"]

    retrieved: Final = await litellm.afile_retrieve(
        file_id="file-abc123", custom_llm_provider="openai", api_key="fake-key"
    )
    assert retrieve_route.called
    assert retrieved.filename == "mydata.jsonl"

    content: Final = await litellm.afile_content(
        file_id="file-abc123", custom_llm_provider="openai", api_key="fake-key"
    )
    assert content_route.called
    assert content.content == _FILE_BODY

    deleted: Final = await litellm.afile_delete(
        file_id="file-abc123", custom_llm_provider="openai", api_key="fake-key"
    )
    assert delete_route.called
    assert deleted.deleted is True
