import base64
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc

_PDF_BYTES: Final = (
    b"%PDF-1.1\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF"
)
_PDF_B64: Final = base64.b64encode(_PDF_BYTES).decode()


def test_pdf_document_block_maps_to_input_file_on_bridge(gateway: Gateway) -> None:
    request_body: Final = {**cc.frontier_request(f"cache-bust-{uuid.uuid4().hex}", "high", 64000), "stream": False}
    doc_text: Final = f"What is on page one? {uuid.uuid4().hex}"
    request_body["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "data": _PDF_B64, "media_type": "application/pdf"},
                    "title": "dot.pdf",
                },
                {"type": "text", "text": doc_text},
            ],
        }
    ]

    def respond(request: Request) -> Reply:
        assert request.target == "/responses", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        user_msg: Final = body["input"][0]
        assert user_msg["content"][0] == {
            "type": "input_file",
            "filename": "dot.pdf",
            "file_data": f"data:application/pdf;base64,{_PDF_B64}",
        }, user_msg
        assert user_msg["content"][1] == {"type": "input_text", "text": doc_text}, user_msg
        return Reply(
            body=cc.responses_completed(
                "doc",
                cc.OPENAI_BACKEND,
                (
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Page one.", "annotations": []}],
                    },
                ),
                {"input_tokens": 41, "output_tokens": 3, "total_tokens": 44},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
