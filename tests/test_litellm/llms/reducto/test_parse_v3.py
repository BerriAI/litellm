import json

import litellm
import pytest


def _reducto_parse_response() -> dict:
    return {
        "job_id": "job_123",
        "usage": {"num_pages": 3, "credits": 3},
        "result": {
            "chunks": [
                {
                    "content": "Page 1 block A",
                    "blocks": [
                        {
                            "content": "Page 1 block A",
                            "bbox": {"page": 1},
                            "kind": "text",
                        }
                    ],
                },
                {
                    "content": "Page 2 block A",
                    "blocks": [
                        {
                            "content": "Page 2 block A",
                            "bbox": {"page": 2},
                            "kind": "table",
                        }
                    ],
                },
                {
                    "content": "Page 1 block B",
                    "blocks": [
                        {
                            "content": "Page 1 block B",
                            "bbox": {"page": 1},
                            "kind": "text",
                        }
                    ],
                },
                {
                    "content": "Page 3 block A",
                    "blocks": [
                        {
                            "content": "Page 3 block A",
                            "bbox": {"page": 3},
                            "kind": "figure",
                        }
                    ],
                },
            ]
        },
    }


@pytest.fixture()
def disable_aiohttp_transport():
    original_disable_aiohttp = litellm.disable_aiohttp_transport
    litellm.disable_aiohttp_transport = True
    litellm.in_memory_llm_clients_cache.flush_cache()
    try:
        yield
    finally:
        litellm.disable_aiohttp_transport = original_disable_aiohttp
        litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_parse_v3_file_upload_and_response_mapping(
    disable_aiohttp_transport, respx_mock
):
    upload_route = respx_mock.post("https://platform.reducto.ai/upload").respond(
        json={"file_id": "reducto://uploaded.pdf"}
    )
    parse_route = respx_mock.post("https://platform.reducto.ai/parse").respond(
        json=_reducto_parse_response()
    )

    response = await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"%PDF-1.4 reducto",
            "mime_type": "application/pdf",
        },
        api_key="test-key",
        api_base="https://platform.reducto.ai",
        formatting={"table_output_format": "html"},
        retrieval={"chunk_mode": "section"},
        settings={"ocr_system": "standard"},
    )

    assert upload_route.called
    assert parse_route.called
    assert len(upload_route.calls) == 1
    assert len(parse_route.calls) == 1

    upload_request = upload_route.calls[0].request
    assert upload_request.headers["authorization"] == "Bearer test-key"
    assert "application/json" not in upload_request.headers["content-type"]
    upload_body = upload_request.read()
    assert b'filename="document"' in upload_body
    assert b"application/pdf" in upload_body

    parse_request_body = json.loads(parse_route.calls[0].request.read())
    assert parse_request_body["input"] == "reducto://uploaded.pdf"
    assert parse_request_body["formatting"] == {"table_output_format": "html"}
    assert parse_request_body["retrieval"] == {"chunk_mode": "section"}
    assert parse_request_body["settings"] == {"ocr_system": "standard"}

    assert response.usage_info is not None
    assert response.usage_info.credits == 3
    assert response.usage_info.pages_processed == 3
    assert len(response.pages) == 3
    assert response.pages[0].index == 0
    assert response.pages[0].markdown == "Page 1 block A\n\nPage 1 block B"
    assert getattr(response.pages[0], "blocks")[0]["bbox"]["page"] == 1
    assert response.pages[1].markdown == "Page 2 block A"
    assert response.pages[2].markdown == "Page 3 block A"
    assert response._hidden_params["reducto_raw"]["usage"]["credits"] == 3


@pytest.mark.asyncio
async def test_parse_v3_reducto_id_passthrough_skips_upload(
    disable_aiohttp_transport, respx_mock
):
    upload_route = respx_mock.post("https://platform.reducto.ai/upload").respond(
        json={"file_id": "reducto://should-not-upload.pdf"}
    )
    parse_route = respx_mock.post("https://platform.reducto.ai/parse").respond(
        json=_reducto_parse_response()
    )

    response = await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "document_url",
            "document_url": "reducto://already-uploaded.pdf",
        },
        api_key="test-key",
        api_base="https://platform.reducto.ai",
        retrieval={"chunk_mode": "section"},
    )

    assert not upload_route.called
    assert parse_route.called
    parse_request_body = json.loads(parse_route.calls[0].request.read())
    assert parse_request_body["input"] == "reducto://already-uploaded.pdf"
    assert parse_request_body["retrieval"]["chunk_mode"] == "section"
    assert response.pages[0].markdown.startswith("Page 1 block A")
