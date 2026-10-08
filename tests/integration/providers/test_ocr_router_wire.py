import json
from typing import Final

import pytest

from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

_DOCUMENT_URL: Final = "https://example.com/doc.pdf"
_OCR_COST_PER_PAGE: Final = 0.0125
_MISTRAL_OCR_BODY: Final = json.dumps(
    {
        "model": "mistral-ocr-latest",
        "pages": [{"index": 0, "markdown": "Test PDF File"}],
        "usage_info": {"pages_processed": 1, "doc_size_bytes": 1024},
    }
).encode()


def _mistral_ocr_peer(request: Request) -> Reply:
    return Reply(body=_MISTRAL_OCR_BODY)


def test_router_aocr_routes_to_mistral_and_logs_spend(gateway: Gateway) -> None:
    with wire_server(_mistral_ocr_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest",
            api_base=wire.url,
            api_key="fake-mistral-key",
            ocr_cost_per_page=_OCR_COST_PER_PAGE,
        )
        response: Final = gateway.request(
            "POST", "/v1/ocr", {"model": model, "document": {"type": "document_url", "document_url": _DOCUMENT_URL}}
        )
        assert response.status_code == 200, response.text
        upstream: Final = wire.drain()
        assert len(upstream) == 1, upstream
        assert (upstream[0].method, upstream[0].target) == ("POST", "/v1/ocr"), upstream[0]
        sent: Final = json.loads(upstream[0].body)
        assert sent["model"] == "mistral-ocr-latest", sent
        assert sent["document"]["type"] == "document_url", sent
        assert sent["document"]["document_url"] == _DOCUMENT_URL, sent
        payload: Final = response.json()
        assert payload["object"] == "ocr", payload
        assert payload["model"] == model, payload
        assert [page["index"] for page in payload["pages"]] == [0], payload
        assert payload["pages"][0]["markdown"] == "Test PDF File", payload
        assert payload["usage_info"]["pages_processed"] == len(payload["pages"]), payload
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_OCR_COST_PER_PAGE), response.headers

        request_id: Final = string_value(response.headers["x-litellm-call-id"])
        rows: Final = eventually(
            lambda: read_rows(
                "SELECT status, call_type, custom_llm_provider, model, model_group, spend, prompt_tokens, "
                'completion_tokens, total_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (request_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "success", row
        assert row["call_type"] == "aocr", row
        assert row["custom_llm_provider"] == "mistral", row
        assert row["model"] == "mistral/mistral-ocr-latest", row
        assert row["model_group"] == model, row
        assert float(row["spend"]) == pytest.approx(_OCR_COST_PER_PAGE), row
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (0, 0, 0), row
