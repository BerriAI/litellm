import json
import pytest

import litellm
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

pytestmark = pytest.mark.requires_rust_extension


def _reply(body: object) -> Reply:
    return Reply(body=json.dumps(body).encode())


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
async def test_parse_v3_file_upload_and_response_mapping(disable_aiohttp_transport, reducto_server: SharedProvider):
    provider_response = _reducto_parse_response()
    reducto_server.expect(_reply({"file_id": "reducto://uploaded.pdf"}), _reply(provider_response))

    response = await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "file",
            "file": b"%PDF-1.4 reducto",
            "mime_type": "application/pdf",
        },
        api_key="test-key",
        api_base=reducto_server.wire.url,
        formatting={"table_output_format": "html"},
        retrieval={"chunk_mode": "section"},
        settings={"ocr_system": "standard"},
        req_format="native",
    )

    requests = reducto_server.received()
    assert len(requests) == 2
    upload_request, parse_request = requests
    assert upload_request.target == "/upload"
    assert parse_request.target == "/parse"
    assert upload_request.headers["authorization"] == "Bearer test-key"
    assert "application/json" not in upload_request.headers["content-type"]
    upload_body = upload_request.body
    assert b'filename="document"' in upload_body
    assert b"application/pdf" in upload_body

    parse_request_body = json.loads(parse_request.body)
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
    assert response.get_provider_native_response() == provider_response


@pytest.mark.asyncio
async def test_parse_v3_reducto_id_passthrough_skips_upload(disable_aiohttp_transport, reducto_server: SharedProvider):
    reducto_server.expect(_reply(_reducto_parse_response()))

    response = await litellm.aocr(
        model="reducto/parse-v3",
        document={
            "type": "document_url",
            "document_url": "reducto://already-uploaded.pdf",
        },
        api_key="test-key",
        api_base=reducto_server.wire.url,
        retrieval={"chunk_mode": "section"},
    )

    requests = reducto_server.received()
    assert len(requests) == 1
    parse_request = requests[0]
    assert parse_request.target == "/parse"
    parse_request_body = json.loads(parse_request.body)
    assert parse_request_body["input"] == "reducto://already-uploaded.pdf"
    assert parse_request_body["retrieval"]["chunk_mode"] == "section"
    assert response.pages[0].markdown.startswith("Page 1 block A")


@pytest.mark.asyncio
async def test_unknown_model_uses_current_protocol_without_local_rejection(
    disable_aiohttp_transport, reducto_server: SharedProvider
):
    reducto_server.expect(_reply(_reducto_parse_response()))

    response = await litellm.aocr(
        model="reducto/future-parse-model",
        document={
            "type": "document_url",
            "document_url": "reducto://already-uploaded.pdf",
        },
        api_key="test-key",
        api_base=reducto_server.wire.url,
    )

    requests = reducto_server.received()
    assert len(requests) == 1
    assert requests[0].target == "/parse"
    assert json.loads(requests[0].body) == {"input": "reducto://already-uploaded.pdf"}
    assert response.model == "future-parse-model"
