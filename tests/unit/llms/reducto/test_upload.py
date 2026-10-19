import json
import os
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import litellm
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

pytestmark = pytest.mark.requires_rust_extension


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
async def test_parse_v3_rejects_plain_http_urls(disable_aiohttp_transport, reducto_server: SharedProvider):
    with pytest.raises(litellm.BadRequestError, match="upload the file first"):
        await litellm.aocr(
            model="reducto/parse-v3",
            document={
                "type": "document_url",
                "document_url": "https://example.com/document.pdf",
            },
            api_key="test-key",
            api_base=reducto_server.wire.url,
        )
    requests = reducto_server.received()
    assert requests == ()


@pytest.mark.asyncio
async def test_parse_v3_image_data_uri_upload_uses_image_mime(
    disable_aiohttp_transport, reducto_server: SharedProvider
):
    reducto_server.expect(
        Reply(body=json.dumps({"file_id": "reducto://uploaded-image.png"}).encode()),
        Reply(
            body=json.dumps(
                {
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
            ).encode()
        ),
    )

    response = await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"\x89PNG\r\n\x1a\npng",
            "mime_type": "image/png",
        },
        api_key="programmatic-key",
        api_base=f"{reducto_server.wire.url}/",
    )

    requests = reducto_server.received()
    assert len(requests) == 2
    upload_request, parse_request = requests
    assert upload_request.target == "/upload"
    assert parse_request.target == "/parse"
    assert upload_request.headers["authorization"] == "Bearer programmatic-key"
    assert b"image/png" in upload_request.body

    assert json.loads(parse_request.body)["input"] == "reducto://uploaded-image.png"
    assert response.pages[0].markdown == "Image OCR"


@pytest.mark.asyncio
async def test_parse_v3_uses_programmatic_api_key_over_env(
    disable_aiohttp_transport, reducto_server: SharedProvider
):
    reducto_server.expect(
        Reply(body=json.dumps({"file_id": "reducto://uploaded.pdf"}).encode()),
        Reply(
            body=json.dumps(
                {
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
            ).encode()
        ),
    )

    await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"%PDF-1.4 auth",
            "mime_type": "application/pdf",
        },
        api_key="passed-key",
        api_base=reducto_server.wire.url,
    )

    requests = reducto_server.received()
    assert len(requests) == 2
    assert requests[0].headers["authorization"] == "Bearer passed-key"
    assert requests[1].headers["authorization"] == "Bearer passed-key"
