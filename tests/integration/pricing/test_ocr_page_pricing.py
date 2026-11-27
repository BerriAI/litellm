import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse


@pytest.mark.covers("pricing.ocr.annotation_pages_billed_at_annotation_rate")
def test_ocr_annotation_pages_are_billed_at_annotation_cost_per_page(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"ocr-annotation-{uuid.uuid4().hex[:12]}"
        handle: Final = register_scenario(
            scenario_id,
            JsonResponse(
                content_type="application/json",
                body={
                    "pages": [{"index": index, "markdown": f"page {index}"} for index in range(3)],
                    "model": "integration-ocr",
                    "document_annotation": '{"title": "annotated"}',
                    "usage_info": {"pages_processed": 3, "pages_processed_annotation": 2, "doc_size_bytes": 4096},
                },
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model=f"mistral/integration-ocr-{scenario_id}",
            api_base=f"{handle.api_base()}/v1",
            ocr_cost_per_page=0.002,
            annotation_cost_per_page=0.01,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/ocr",
            {
                "model": model,
                "document": {"type": "document_url", "document_url": "https://example.com/annotated.pdf"},
                "document_annotation_format": {"type": "json_schema", "json_schema": {"name": "title"}},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["usage_info"] == {
            "pages_processed": 3,
            "pages_processed_annotation": 2,
            "credits": None,
            "doc_size_bytes": 4096,
        }, response.text
        expected: Final = 3 * 0.002 + 2 * 0.01
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected), response.text
        request_id: Final = string_value(response.headers["x-litellm-call-id"])
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(expected)
