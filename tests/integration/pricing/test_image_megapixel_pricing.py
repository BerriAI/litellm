import asyncio
import json
from typing import Final

import openai
import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

import litellm
from litellm import get_model_info
from tests.integration._support.client import Gateway, eventually, object_value, string_value
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
    num_images: int = 1


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
    (
        ("512x512", 0.25),
        ("1024x1024", 1.0),
        ("1040x1024", 1040 * 1024 / 1_048_576),
        ("1920x1080", 1920 * 1080 / 1_048_576),
        ("2048x2048", 4.0),
    ),
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


def _catalog_price(field: str) -> float:
    return _PRICE.validate_python(get_model_info(_MODEL).get(field))


def _catalog_tiered(megapixels: float) -> float:
    return _catalog_price("output_cost_per_first_megapixel") * min(megapixels, 1.0) + _catalog_price(
        "output_cost_per_additional_megapixel"
    ) * max(megapixels - 1.0, 0.0)


def _images_reply(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    count: Final = int(body.get("num_images") or body.get("n") or 1)
    return Reply(body=json.dumps({"data": [{"b64_json": "aW1n"} for _ in range(count)]}).encode())


def _spend_row(request_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, model, status, api_base FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _bill(gateway: Gateway, body: dict[str, JsonValue], **pricing: JsonValue) -> tuple[float, float, bytes]:
    with wire_server(_images_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=str(pricing.pop("model", _MODEL)),
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="preview",
            **pricing,
        )
        response: Final = gateway.request("POST", "/v1/images/generations", {"model": model, **body})
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
        assert len(received) == 1, received
        row: Final = _spend_row(string_value(response.headers["x-litellm-call-id"]))
        return (
            float(response.headers["x-litellm-response-cost"]),
            _PRICE.validate_python(row["spend"]),
            received[0].body,
        )


def _billed_body(gateway: Gateway, body: dict[str, JsonValue], **pricing: JsonValue) -> _Billed:
    response_cost, spend, outbound = _bill(gateway, body, **pricing)
    return _Billed(response_cost=response_cost, spend=spend, outbound=_FluxRequest.model_validate_json(outbound))


def test_openai_sdk_sync_and_async_clients_are_billed_by_megapixel(gateway: Gateway) -> None:
    with wire_server(_images_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
        )
        base_url: Final = f"{gateway.client.base_url}/v1"
        sync_client: Final = openai.OpenAI(base_url=base_url, api_key=gateway.key, max_retries=0)
        sync_raw: Final = sync_client.images.with_raw_response.generate(
            model=model, prompt=_PROMPT, size="1024x1024", n=1
        )
        async_client: Final = openai.AsyncOpenAI(base_url=base_url, api_key=gateway.key, max_retries=0)
        async_raw: Final = asyncio.run(
            async_client.images.with_raw_response.generate(
                model=model,
                prompt=_PROMPT,
                size="1792x1024",
                n=1,
            )
        )
        outbound: Final = tuple(_FluxRequest.model_validate_json(request.body) for request in wire.drain())
    assert sorted((request.width, request.height) for request in outbound) == [(1024, 1024), (1792, 1024)]
    for raw, megapixels in ((sync_raw, 1.0), (async_raw, 1792 * 1024 / 1_048_576)):
        assert len(raw.parse().data or ()) == 1
        row: Final = _spend_row(raw.headers["x-litellm-call-id"])
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(_catalog_tiered(megapixels)), row
        assert _PRICE.validate_python(row["spend"]) == pytest.approx(_catalog_tiered(megapixels)), row


def test_explicit_width_and_height_without_size_are_billed_by_megapixel(gateway: Gateway) -> None:
    billed: Final = _billed_body(gateway, {"prompt": _PROMPT, "width": 2048, "height": 2048})

    assert billed.outbound == _FluxRequest(width=2048, height=2048, num_images=1), billed
    assert billed.response_cost == pytest.approx(_catalog_tiered(4.0)), billed
    assert billed.spend == pytest.approx(billed.response_cost), billed


def test_string_width_and_height_override_size_for_the_bill(gateway: Gateway) -> None:
    billed: Final = _billed_body(gateway, {"prompt": _PROMPT, "size": "1024x1024", "width": "2048", "height": "2048"})

    assert (billed.outbound.width, billed.outbound.height) == (2048, 2048), billed
    assert billed.response_cost == pytest.approx(_catalog_tiered(4.0)), billed
    assert billed.spend == pytest.approx(billed.response_cost), billed


@pytest.mark.parametrize(
    ("deployment_prices", "expected"),
    (
        ({"output_cost_per_first_megapixel": 0.05}, "first-only"),
        ({"output_cost_per_additional_megapixel": 0.02}, "additional-only"),
    ),
)
def test_partial_deployment_megapixel_price_keeps_the_other_catalog_tier(
    gateway: Gateway, deployment_prices: dict[str, JsonValue], expected: str
) -> None:
    first: Final = 0.05 if expected == "first-only" else _catalog_price("output_cost_per_first_megapixel")
    additional: Final = (
        0.02 if expected == "additional-only" else _catalog_price("output_cost_per_additional_megapixel")
    )

    billed: Final = _billed_body(gateway, {"prompt": _PROMPT, "size": "2048x2048"}, **deployment_prices)

    assert billed.response_cost == pytest.approx(first + 3 * additional), billed
    assert billed.spend == pytest.approx(billed.response_cost), billed


def test_deployment_flat_image_price_still_bills_per_image(gateway: Gateway) -> None:
    billed: Final = _billed_body(gateway, {"prompt": _PROMPT, "size": "2048x2048", "n": 2}, output_cost_per_image=0.07)

    assert billed.response_cost == pytest.approx(2 * 0.07), billed
    assert billed.spend == pytest.approx(billed.response_cost), billed


def test_deployment_per_pixel_price_on_flux_2_pro_stays_shadowed_by_the_catalog_flat_price(gateway: Gateway) -> None:
    billed: Final = _billed_body(gateway, {"prompt": _PROMPT, "size": "1024x1024"}, input_cost_per_pixel=1e-8)

    assert billed.response_cost == pytest.approx(_catalog_price("output_cost_per_image")), billed
    assert billed.spend == pytest.approx(billed.response_cost), billed


@pytest.mark.parametrize("sibling", ("azure_ai/FLUX-1.1-pro", "azure_ai/FLUX.2-flex"))
def test_sibling_flux_models_keep_their_catalog_pricing(gateway: Gateway, sibling: str) -> None:
    catalog: Final = litellm.model_cost[sibling]
    flat: Final = catalog.get("output_cost_per_image")

    response_cost, spend, _ = _bill(gateway, {"prompt": _PROMPT, "size": "1024x1024"}, model=sibling)

    expected: Final = (
        _PRICE.validate_python(flat) if flat else 1024 * 1024 * _PRICE.validate_python(catalog["input_cost_per_pixel"])
    )
    assert response_cost == pytest.approx(expected), (sibling, response_cost)
    assert spend == pytest.approx(response_cost), (sibling, spend)


def test_size_auto_is_billed_as_the_default_1024_square_first_megapixel(gateway: Gateway) -> None:
    with wire_server(_images_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
        )
        response: Final = gateway.request(
            "POST", "/v1/images/generations", {"model": model, "prompt": _PROMPT, "size": "auto"}
        )
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(received) == 1 and "width" not in json.loads(received[0].body), received
    row: Final = _spend_row(string_value(response.headers["x-litellm-call-id"]))
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_catalog_tiered(1.0)), row
    assert _PRICE.validate_python(row["spend"]) == pytest.approx(_catalog_tiered(1.0)), row


@pytest.mark.parametrize("size", ("banana", 1024, ["1024", "1024"], "", "9" * 5000 + "x1024"))
def test_malformed_size_is_rejected_before_the_upstream_and_the_proxy_keeps_serving(
    gateway: Gateway, size: JsonValue
) -> None:
    with wire_server(_images_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
        )
        rejected: Final = gateway.request(
            "POST", "/v1/images/generations", {"model": model, "prompt": _PROMPT, "size": size}
        )
        rejected_upstream: Final = wire.drain()
        healthy: Final = gateway.request(
            "POST", "/v1/images/generations", {"model": model, "prompt": _PROMPT, "size": "1024x1024"}
        )
        healthy_upstream: Final = wire.drain()
    assert 400 <= rejected.status_code < 500, rejected.text
    assert "error" in rejected.json(), rejected.text
    assert rejected_upstream == (), rejected_upstream
    assert healthy.status_code == 200, healthy.text
    assert len(healthy_upstream) == 1, healthy_upstream
    assert float(healthy.headers["x-litellm-response-cost"]) == pytest.approx(_catalog_tiered(1.0)), healthy.headers


@pytest.mark.parametrize("status", (401, 500))
def test_upstream_error_reaches_the_caller_and_logs_a_zero_spend_failure(gateway: Gateway, status: int) -> None:
    def fail(request: Request) -> Reply:
        return Reply(status=status, body=json.dumps({"error": {"code": str(status), "message": "scripted"}}).encode())

    with wire_server(fail) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview", num_retries=0
        )
        response: Final = gateway.request(
            "POST", "/v1/images/generations", {"model": model, "prompt": _PROMPT, "size": "2048x2048"}
        )
        received: Final = wire.drain()
    assert response.status_code == status, response.text
    assert "scripted" in response.text, response.text
    assert len(received) == 1, received
    row: Final = _spend_row(string_value(response.headers["x-litellm-call-id"]))
    assert row["status"] == "failure", row
    assert _PRICE.validate_python(row["spend"]) == 0.0, row


def test_unauthenticated_image_request_never_reaches_the_upstream(gateway: Gateway) -> None:
    with wire_server(_images_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
        )
        response: Final = gateway.request(
            "POST", "/v1/images/generations", {"model": model, "prompt": _PROMPT, "size": "2048x2048"}, key="sk-bogus"
        )
        received: Final = wire.drain()
    assert response.status_code == 401, response.text
    assert received == (), received


def test_identical_requests_each_write_one_row_with_the_same_spend(gateway: Gateway) -> None:
    with wire_server(_images_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
        )
        body: Final[dict[str, JsonValue]] = {"model": model, "prompt": _PROMPT, "size": "1920x1080", "n": 3}
        responses: Final = tuple(gateway.request("POST", "/v1/images/generations", body) for _ in range(2))
        received: Final = wire.drain()
    assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
    assert [_FluxRequest.model_validate_json(request.body).num_images for request in received] == [3, 3]
    ids: Final = tuple(string_value(response.headers["x-litellm-call-id"]) for response in responses)
    assert len(set(ids)) == 2, ids
    expected: Final = 3 * _catalog_tiered(1920 * 1080 / 1_048_576)
    for request_id in ids:
        assert _PRICE.validate_python(_spend_row(request_id)["spend"]) == pytest.approx(expected), request_id


def test_deployment_megapixel_prices_round_trip_through_model_info(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base="http://127.0.0.1:9",
            api_key="synthetic-azure-key",
            output_cost_per_first_megapixel=0.05,
            output_cost_per_additional_megapixel=0.02,
        )
        entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    matching: Final = tuple(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model)
    assert len(matching) == 1, matching
    info: Final = object_value(matching[0]["model_info"])
    assert (info["output_cost_per_first_megapixel"], info["output_cost_per_additional_megapixel"]) == (0.05, 0.02), info


def test_azure_ai_passthrough_flux_2_pro_is_billed_by_megapixel(gateway: Gateway) -> None:
    with wire_server(_images_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL, api_base=wire.url, api_key="synthetic-azure-key", api_version="preview"
        )
        response: Final = gateway.request(
            "POST",
            f"/azure_ai/{model}/providers/blackforestlabs/v1/flux-2-pro",
            {"prompt": _PROMPT, "width": 2048, "height": 2048, "num_images": 1},
            params={"api-version": "preview"},
        )
        received: Final = wire.drain()
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
    assert response.status_code == 200, response.text
    assert [request.target for request in received] == [_TARGET], received
    assert _PRICE.validate_python(rows[0]["spend"]) == pytest.approx(_catalog_tiered(4.0)), rows
