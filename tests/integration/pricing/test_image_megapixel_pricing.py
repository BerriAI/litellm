import json
from typing import Final

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm import get_model_info
from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "azure_ai/flux.2-pro"
_PROMPT: Final = "a lighthouse at dusk"
_TARGET: Final = "/providers/blackforestlabs/v1/flux-2-pro?api-version=preview"
_PRICE: Final = TypeAdapter(float)


class _Images(BaseModel):
    data: list[dict[str, JsonValue]]


class _FluxRequest(BaseModel):
    width: int
    height: int
    num_images: int


class _Billed(BaseModel):
    response_cost: float
    spend: float
    outbound: _FluxRequest


def _generate(gateway: Gateway, size: str, n: int, **pricing: JsonValue) -> _Billed:
    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps({"data": [{"b64_json": "aW1n"} for _ in range(n)]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview", **pricing
        )
        response: Final = gateway.request(
            "POST", "/v1/images/generations", {"model": model, "prompt": _PROMPT, "n": n, "size": size}
        )
        assert response.status_code == 200, response.text
        assert len(_Images.model_validate_json(response.content).data) == n, response.text
        received: Final = wire.drain()
        assert [(request.method, request.target) for request in received] == [("POST", _TARGET)]
        request_id: Final = string_value(response.headers["x-litellm-call-id"])
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        return _Billed(
            response_cost=float(response.headers["x-litellm-response-cost"]),
            spend=_PRICE.validate_python(rows[0]["spend"]),
            outbound=_FluxRequest.model_validate_json(received[0].body),
        )


def test_deployment_first_and_additional_megapixel_prices_bill_each_image(gateway: Gateway) -> None:
    billed: Final = _generate(
        gateway,
        "2048x2048",
        2,
        output_cost_per_first_megapixel=0.05,
        output_cost_per_additional_megapixel=0.02,
    )

    assert billed.outbound == _FluxRequest(width=2048, height=2048, num_images=2), billed
    expected: Final = 2 * (0.05 + 3 * 0.02)
    assert billed.response_cost == pytest.approx(expected), billed
    assert billed.spend == pytest.approx(expected), billed


@pytest.mark.parametrize(
    ("size", "megapixels"),
    (("512x512", 0.25), ("1024x1024", 1.0), ("1920x1080", 1920 * 1080 / 1_048_576), ("2048x2048", 4.0)),
)
def test_catalog_flux_2_pro_bills_first_then_additional_fractional_megapixels(
    gateway: Gateway, size: str, megapixels: float
) -> None:
    catalog: Final = get_model_info(_MODEL)
    first: Final = _PRICE.validate_python(catalog.get("output_cost_per_first_megapixel"))
    additional: Final = _PRICE.validate_python(catalog.get("output_cost_per_additional_megapixel"))

    billed: Final = _generate(gateway, size, 1)

    expected: Final = first * min(megapixels, 1.0) + additional * max(megapixels - 1.0, 0.0)
    assert billed.response_cost == pytest.approx(expected), billed
    assert billed.spend == pytest.approx(expected), billed
    assert billed.response_cost != pytest.approx(catalog.get("output_cost_per_image")), billed
