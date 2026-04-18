import json
import os

import litellm
import pytest


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
