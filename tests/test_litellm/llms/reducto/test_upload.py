import json
import os
from unittest.mock import AsyncMock, Mock

import httpx
import litellm
import pytest

from litellm.llms.reducto.common import (
    extract_file_id_or_bytes,
    upload_bytes_async,
    upload_bytes_sync,
)


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
async def test_parse_v3_rejects_plain_http_urls(disable_aiohttp_transport):
    with pytest.raises(litellm.BadRequestError, match="upload the file first"):
        await litellm.aocr(
            model="reducto/parse-v3",
            document={
                "type": "document_url",
                "document_url": "https://example.com/document.pdf",
            },
            api_key="test-key",
            api_base="https://platform.reducto.ai",
        )


@pytest.mark.asyncio
async def test_parse_v3_image_data_uri_upload_uses_image_mime(
    disable_aiohttp_transport, respx_mock
):
    upload_route = respx_mock.post("https://custom.reducto.test/upload").respond(
        json={"file_id": "reducto://uploaded-image.png"}
    )
    parse_route = respx_mock.post("https://custom.reducto.test/parse").respond(
        json={
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

    response = await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"\x89PNG\r\n\x1a\npng",
            "mime_type": "image/png",
        },
        api_key="programmatic-key",
        api_base="https://custom.reducto.test/",
    )

    assert upload_route.called
    assert parse_route.called
    upload_request = upload_route.calls[0].request
    assert upload_request.headers["authorization"] == "Bearer programmatic-key"
    assert b"image/png" in upload_request.read()

    parse_request_body = json.loads(parse_route.calls[0].request.read())
    assert parse_request_body["input"] == "reducto://uploaded-image.png"
    assert response.pages[0].markdown == "Image OCR"


@pytest.mark.asyncio
async def test_parse_v3_uses_programmatic_api_key_over_env(
    disable_aiohttp_transport, respx_mock
):
    upload_route = respx_mock.post("https://platform.reducto.ai/upload").respond(
        json={"file_id": "reducto://uploaded.pdf"}
    )
    parse_route = respx_mock.post("https://platform.reducto.ai/parse").respond(
        json={
            "usage": {"num_pages": 1, "credits": 1},
            "result": {
                "chunks": [
                    {
                        "content": "Programmatic auth",
                        "blocks": [
                            {"content": "Programmatic auth", "bbox": {"page": 1}}
                        ],
                    }
                ]
            },
        }
    )

    await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"%PDF-1.4 auth",
            "mime_type": "application/pdf",
        },
        api_key="passed-key",
        api_base="https://platform.reducto.ai",
    )

    assert upload_route.calls[0].request.headers["authorization"] == "Bearer passed-key"
    assert parse_route.calls[0].request.headers["authorization"] == "Bearer passed-key"


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
