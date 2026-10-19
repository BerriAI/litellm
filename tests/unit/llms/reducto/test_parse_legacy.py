import json
import pytest

import litellm
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

pytestmark = pytest.mark.requires_rust_extension


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
async def test_parse_legacy_wraps_enhance_under_options(disable_aiohttp_transport, reducto_server: SharedProvider):
    reducto_server.expect(
        Reply(body=json.dumps({"file_id": "reducto://legacy.pdf"}).encode()),
        Reply(
            body=json.dumps(
                {
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
            ).encode()
        ),
    )

    response = await litellm.aocr(
        model="reducto/parse-legacy",
        document={
            "type": "file",
            "file": b"%PDF-1.4 legacy",
            "mime_type": "application/pdf",
        },
        api_key="legacy-key",
        api_base=reducto_server.wire.url,
        enhance={"agentic": [{"type": "table"}]},
    )

    requests = reducto_server.received()
    assert len(requests) == 2
    upload_request, parse_request = requests
    assert upload_request.target == "/upload"
    assert parse_request.target == "/parse"
    request_body = json.loads(parse_request.body)
    assert request_body == {
        "document_url": "reducto://legacy.pdf",
        "options": {"enhance": {"agentic": [{"type": "table"}]}},
    }
    assert response.pages[0].markdown == "Legacy parse"
