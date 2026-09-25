import json
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, wire_server

_PIXELS_PER_MEGAPIXEL: Final = 1024 * 1024


@pytest.mark.parametrize("input_cost_per_pixel", (1.5e-07, 0.0))
def test_flux2_edit_bills_azure_reported_megapixels_at_the_deployment_rate(
    gateway: Gateway, input_cost_per_pixel: float
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        return Reply(
            body=json.dumps(
                {"data": [{"b64_json": "aW1n"}], "request_meta": {"input_mp": 0.39, "output_mp": 1.0}}
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure_ai/FLUX.2-flex",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            input_cost_per_pixel=input_cost_per_pixel,
        )
        response: Final = gateway.request_multipart(
            "/v1/images/edits",
            {"model": model, "prompt": "add a hat", "size": "1024x1024"},
            {"image": ("reference.png", b"reference", "image/png")},
        )
        assert response.status_code == 200, response.text
        expected: Final = input_cost_per_pixel * _PIXELS_PER_MEGAPIXEL * (0.39 + 1.0)
        if expected:
            assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected), response.text
        else:
            assert response.headers.get("x-litellm-response-cost") in (None, "0", "0.0"), response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (call_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(expected), response.text
