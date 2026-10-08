import json
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

_MISTRAL_OCR_BODY: Final = json.dumps(
    {
        "model": "mistral-ocr-latest",
        "pages": [{"index": 0, "markdown": "Test PDF File"}],
        "usage_info": {"pages_processed": 1, "doc_size_bytes": 1024},
    }
).encode()


def test_router_aocr_routes_to_mistral_and_logs_spend(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/ocr", request.target
        body: Final = json.loads(request.body)
        assert body["model"] == "mistral-ocr-latest"
        assert body["document"]["document_url"] == "https://example.com/doc.pdf"
        return Reply(body=_MISTRAL_OCR_BODY)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest", api_base=wire.url, api_key="fake-mistral-key"
        )
        response: Final = gateway.request(
            "POST", "/v1/ocr", {"model": model, "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"}}
        )
        assert response.status_code == 200, response.text
        payload: Final = response.json()
        assert payload["object"] == "ocr"
        assert [page["index"] for page in payload["pages"]] == [0]
        assert "test pdf file" in " ".join(page["markdown"] for page in payload["pages"]).lower()

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, call_type, custom_llm_provider, model, spend FROM "LiteLLM_SpendLogs" WHERE model=%s',
                ("mistral/mistral-ocr-latest",),
            ),
            lambda values: len(values) == 1,
        )
        row: Final = rows[0]
        assert row["status"] == "success"
        assert row["call_type"] == "aocr"
        assert row["custom_llm_provider"] == "mistral"
        assert row["model"] == "mistral/mistral-ocr-latest"
        assert float(row["spend"]) > 0
