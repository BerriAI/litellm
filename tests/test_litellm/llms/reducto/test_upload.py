import os
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import litellm
from litellm.llms.reducto.common import (
    extract_file_id_or_bytes,
    upload_bytes_async,
    upload_bytes_sync,
)
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


@pytest.fixture()
def disable_aiohttp_transport(monkeypatch):
    original_disable_aiohttp = litellm.disable_aiohttp_transport
    litellm.disable_aiohttp_transport = True
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.setenv("REDUCTO_API_KEY", "env-reducto-key")
    try:
        yield
    finally:
        litellm.disable_aiohttp_transport = original_disable_aiohttp
        litellm.in_memory_llm_clients_cache.flush_cache()
        os.environ.pop("REDUCTO_API_KEY", None)


@pytest.mark.asyncio
async def test_parse_v3_rejects_plain_http_urls(disable_aiohttp_transport, reducto_server: RecordingServer):
    reducto_server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError, match="upload the file first"):
        await litellm.aocr(
            model="reducto/parse-v3",
            document={
                "type": "document_url",
                "document_url": "https://example.com/document.pdf",
            },
            api_key="test-key",
            api_base=reducto_server.base_url,
        )


@pytest.mark.asyncio
async def test_parse_v3_image_data_uri_upload_uses_image_mime(
    disable_aiohttp_transport, reducto_server: RecordingServer
):
    reducto_server.expected_requests = 2
    reducto_server.enqueue(ResponseSpec(body={"file_id": "reducto://uploaded-image.png"}))
    reducto_server.enqueue(
        ResponseSpec(
            body={
                "usage": {"num_pages": 1, "credits": 1},
                "result": {
                    "chunks": [
                        {
                            "content": "Image OCR",
                            "blocks": [{"content": "Image OCR", "bbox": {"page": 1}}],
                        }
                    ]
                },
            }
        )
    )

    response = await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"\x89PNG\r\n\x1a\npng",
            "mime_type": "image/png",
        },
        api_key="programmatic-key",
        api_base=f"{reducto_server.base_url}/",
    )

    upload_request, parse_request = reducto_server.requests
    assert upload_request.path == "/upload"
    assert parse_request.path == "/parse"
    assert upload_request.headers["authorization"] == "Bearer programmatic-key"
    assert b"image/png" in upload_request.raw_body

    assert isinstance(parse_request.body, dict)
    assert parse_request.body["input"] == "reducto://uploaded-image.png"
    assert response.pages[0].markdown == "Image OCR"


@pytest.mark.asyncio
async def test_parse_v3_uses_programmatic_api_key_over_env(disable_aiohttp_transport, reducto_server: RecordingServer):
    reducto_server.expected_requests = 2
    reducto_server.enqueue(ResponseSpec(body={"file_id": "reducto://uploaded.pdf"}))
    reducto_server.enqueue(
        ResponseSpec(
            body={
                "usage": {"num_pages": 1, "credits": 1},
                "result": {
                    "chunks": [
                        {
                            "content": "Programmatic auth",
                            "blocks": [
                                {
                                    "content": "Programmatic auth",
                                    "bbox": {"page": 1},
                                }
                            ],
                        }
                    ]
                },
            }
        )
    )

    await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"%PDF-1.4 auth",
            "mime_type": "application/pdf",
        },
        api_key="passed-key",
        api_base=reducto_server.base_url,
    )

    assert reducto_server.requests[0].headers["authorization"] == "Bearer passed-key"
    assert reducto_server.requests[1].headers["authorization"] == "Bearer passed-key"


def test_upload_bytes_sync_uses_shared_client(monkeypatch):
    captured = {}

    def fake_post(*, url, headers, files, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["files"] = files
        captured["timeout"] = timeout
        return httpx.Response(
            200,
            json={"file_id": "reducto://sync-upload"},
            request=httpx.Request("POST", url),
        )

    sync_post = Mock(side_effect=fake_post)
    monkeypatch.setattr(litellm.module_level_client, "post", sync_post)

    class ForbiddenSyncClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("should not construct")

    monkeypatch.setattr(httpx, "Client", ForbiddenSyncClient)

    file_id = upload_bytes_sync(
        raw_bytes=b"%PDF-1.4 sync",
        mime="application/pdf",
        api_key="sync-key",
        api_base="https://sync.reducto.test/",
    )

    assert file_id == "reducto://sync-upload"
    sync_post.assert_called_once()
    assert captured["url"] == "https://sync.reducto.test/upload"
    assert captured["headers"] == {"Authorization": "Bearer sync-key"}
    assert captured["files"]["file"] == (
        "document",
        b"%PDF-1.4 sync",
        "application/pdf",
    )


@pytest.mark.asyncio
async def test_upload_bytes_async_uses_shared_aclient(monkeypatch):
    captured = {}

    async def fake_post(*, url, headers, files, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["files"] = files
        captured["timeout"] = timeout
        return httpx.Response(
            200,
            json={"file_id": "reducto://async-upload"},
            request=httpx.Request("POST", url),
        )

    async_post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(litellm.module_level_aclient, "post", async_post)

    class ForbiddenAsyncClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("should not construct")

    monkeypatch.setattr(httpx, "AsyncClient", ForbiddenAsyncClient)

    file_id = await upload_bytes_async(
        raw_bytes=b"%PDF-1.4 async",
        mime="application/pdf",
        api_key="async-key",
        api_base="https://async.reducto.test/",
    )

    assert file_id == "reducto://async-upload"
    async_post.assert_awaited_once()
    assert captured["url"] == "https://async.reducto.test/upload"
    assert captured["headers"] == {"Authorization": "Bearer async-key"}
    assert captured["files"]["file"] == (
        "document",
        b"%PDF-1.4 async",
        "application/pdf",
    )


def test_extract_file_id_or_bytes_raises_on_malformed_data_uri():
    with pytest.raises(litellm.BadRequestError, match="Invalid Reducto data URI"):
        extract_file_id_or_bytes("data:application/pdf", model="reducto/parse-v3")

    with pytest.raises(litellm.BadRequestError, match="Invalid Reducto base64 payload"):
        extract_file_id_or_bytes("data:;base64,!!!not-base64", model="reducto/parse-v3")
