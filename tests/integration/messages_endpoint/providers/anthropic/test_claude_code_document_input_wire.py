import base64
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue

_PDF_BYTES: Final = (
    b"%PDF-1.1\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF"
)
_DOC_BLOCK: Final = {
    "type": "document",
    "source": {"type": "base64", "data": base64.b64encode(_PDF_BYTES).decode(), "media_type": "application/pdf"},
    "citations": {"enabled": True},
}


def _cited_stream(identity: str) -> tuple[bytes, ...]:
    return (
        cc.sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": cc.FABLE,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 20, "output_tokens": 1},
                },
            },
        ),
        cc.sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "A page."}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "citations_delta",
                    "citation": {
                        "type": "page_location",
                        "document_index": 0,
                        "document_title": "dot.pdf",
                        "start_page_number": 1,
                        "end_page_number": 1,
                        "cited_text": "Page",
                    },
                },
            },
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        cc.sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 6},
            },
        ),
        cc.sse_frame("message_stop", {"type": "message_stop"}),
    )


def test_base64_pdf_document_with_citations_reaches_anthropic_identical(gateway: Gateway) -> None:
    request_body: Final = cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000)
    request_body["messages"] = [
        {
            "role": "user",
            "content": [
                dict(_DOC_BLOCK),
                {"type": "text", "text": f"What is on page one? {uuid.uuid4().hex}"},
            ],
        }
    ]

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = {**request_body, "model": cc.FABLE}
        assert body == expected, {
            key: (expected.get(key), body.get(key))
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        return Reply(content_type="text/event-stream", chunks=_cited_stream(f"msg_doc_{uuid.uuid4().hex}"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        events: Final = cc.sse_events(response.text)
        citations: Final = [
            data["delta"] for event, data in events if data.get("delta", {}).get("type") == "citations_delta"
        ]
        assert citations == [
            {
                "type": "citations_delta",
                "citation": {
                    "type": "page_location",
                    "document_index": 0,
                    "document_title": "dot.pdf",
                    "start_page_number": 1,
                    "end_page_number": 1,
                    "cited_text": "Page",
                },
            }
        ], citations
        assert len(wire.drain()) == 1
