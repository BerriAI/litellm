import pytest

import litellm
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


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
async def test_parse_legacy_wraps_enhance_under_options(disable_aiohttp_transport, reducto_server: RecordingServer):
    reducto_server.expected_requests = 2
    reducto_server.enqueue(ResponseSpec(body={"file_id": "reducto://legacy.pdf"}))
    reducto_server.enqueue(
        ResponseSpec(
            body={
                "usage": {"num_pages": 1, "credits": 1},
                "result": {
                    "chunks": [
                        {
                            "content": "Legacy parse",
                            "blocks": [
                                {
                                    "content": "Legacy parse",
                                    "bbox": {"page": 1},
                                }
                            ],
                        }
                    ]
                },
            }
        )
    )

    response = await litellm.aocr(
        model="reducto/parse-legacy",
        document={
            "type": "file",
            "file": b"%PDF-1.4 legacy",
            "mime_type": "application/pdf",
        },
        api_key="legacy-key",
        api_base=reducto_server.base_url,
        enhance={"agentic": [{"type": "table"}]},
    )

    upload_request, parse_request = reducto_server.requests
    assert upload_request.path == "/upload"
    assert parse_request.path == "/parse"
    assert isinstance(parse_request.body, dict)
    request_body = parse_request.body
    assert request_body == {
        "document_url": "reducto://legacy.pdf",
        "options": {"enhance": {"agentic": [{"type": "table"}]}},
    }
    assert response.pages[0].markdown == "Legacy parse"
