"""`POST /v1/ocr` on Mistral and Azure AI, served by the Rust OCR runtime behind the Python proxy endpoint.

The proxy endpoint (litellm/proxy/ocr_endpoints/endpoints.py) accepts either a JSON body
`{"model", "document": {"type": "document_url"|"image_url", ...}, <params>}` or multipart form data with a `file`
part, which it turns into a base64 `data:` URL document. It refuses `document.type == "file"` and `reducto://` ids
on the JSON path, empty or unparsable bodies, and multipart bodies without a file or with an empty file, each with
400 and an OpenAI-style error.

Mistral (litellm/rust/crates/llms/src/mistral/ocr/transformation.rs) gets one POST to `<api_base>/v1/ocr` with
`Authorization: Bearer <key>` and the body `{"model", "document", <supported params>}`; params outside the
supported list (`pages`, `include_image_base64`, `image_limit`, `image_min_size`, `bbox_annotation_format`,
`document_annotation_format`, `document_annotation_prompt`, `extract_header`, `extract_footer`, `table_format`,
`confidence_scores_granularity`, `include_blocks`, `id`) are not sent. Azure AI's Mistral document model
(azure_ai/ocr/transformation.rs) reuses the same body at `<api_base>/providers/mistral/azure/ocr`, and only accepts
inline documents, so a remote URL is downloaded by the proxy and sent as a data URL.

Cost (litellm/cost_calculator.py ocr_cost): `ocr_cost_per_credit x credits` when both exist, else
`ocr_cost_per_page x pages_processed + annotation rate x pages_processed_annotation`, where the annotation rate
falls back to the page rate. `test_ocr_router_wire.py`, `test_ocr_multipart_wire.py` and
`pricing/test_ocr_page_pricing.py` already cover the basic Mistral call, multipart data URLs and the
annotation rate.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media2_support import (
    AZURE_KEY,
    MISTRAL_KEY,
    PDF_BYTES,
    data_url,
    error_message,
    error_of,
    json_body,
    json_reply,
    logged_rows,
    mistral_ocr_reply,
    path_of,
    query_of,
    response_cost,
    settled_key_spend,
)

_OCR: Final = "/v1/ocr"
_PAGE_RATE: Final = 0.002


def _document() -> dict[str, str]:
    return {"type": "document_url", "document_url": f"https://docs.invalid/{uuid.uuid4().hex}.pdf"}


def _inline_document() -> dict[str, str]:
    return {"type": "document_url", "document_url": data_url("application/pdf", PDF_BYTES + uuid.uuid4().bytes)}


def _two_pages(_request: Request) -> Reply:
    return mistral_ocr_reply(2)


# ---------------------------------------------------------------- request validation at the proxy


def test_ocr_json_document_of_type_file_is_refused_with_400(gateway: Gateway) -> None:
    """A JSON body names `document: {"type": "file", "file": "/etc/passwd"}`. The proxy must answer 400 with a
    message starting `document type 'file' is not supported through the JSON API` and never call Mistral, since
    a file path would read the proxy's own filesystem."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.request(
            "POST", _OCR, {"model": model, "document": {"type": "file", "file": "/etc/passwd"}}
        )
        assert response.status_code == 400, response.text
        assert error_message(response).startswith("document type 'file' is not supported through the JSON API")
        assert wire.drain() == ()


def test_ocr_reducto_file_id_on_the_json_path_is_refused_with_400(gateway: Gateway) -> None:
    """A JSON body names `document_url` `reducto://abc`. The proxy must answer 400 with a message starting
    `reducto:// file IDs are not accepted through the proxy OCR API` before any provider call, because those ids
    are not scoped to the caller's key."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.request(
            "POST", _OCR, {"model": model, "document": {"type": "document_url", "document_url": "reducto://abc"}}
        )
        assert response.status_code == 400, response.text
        assert error_message(response).startswith("reducto:// file IDs are not accepted through the proxy OCR API")
        assert wire.drain() == ()


def test_ocr_invalid_json_body_is_refused_with_400(gateway: Gateway) -> None:
    """The caller sends `{"model": ` (truncated JSON) with `content-type: application/json`. The proxy must answer
    400 with a message starting `Invalid JSON in request body`.
    UNVERIFIED: auth middleware may reject the unparsable body first with its own 400 message."""
    response: Final = gateway.client.post(
        _OCR,
        content=b'{"model": ',
        headers={"Authorization": f"Bearer {gateway.key}", "content-type": "application/json"},
    )
    assert response.status_code == 400, response.text
    assert "Invalid JSON" in response.text


def test_ocr_multipart_without_a_file_part_is_refused_with_400_naming_the_file_param(gateway: Gateway) -> None:
    """A multipart body carries only `model`. The proxy must answer 400 with `param` `file` and the message
    `Multipart OCR request must include a 'file' field with the document to process`."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.client.post(
            _OCR,
            data={"model": model},
            files={"other": ("x.txt", b"x", "text/plain")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 400, response.text
        assert (
            error_message(response) == "Multipart OCR request must include a 'file' field with the document to process"
        )
        assert error_of(response)["param"] == "file"
        assert wire.drain() == ()


def test_ocr_multipart_with_an_empty_file_is_refused_with_400(gateway: Gateway) -> None:
    """A multipart body uploads a zero byte `file`. The proxy must answer 400 `Uploaded file is empty` with `param`
    `file` and not call Mistral."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.client.post(
            _OCR,
            data={"model": model},
            files={"file": ("empty.pdf", b"", "application/pdf")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 400, response.text
        assert error_message(response) == "Uploaded file is empty"
        assert error_of(response)["param"] == "file"
        assert wire.drain() == ()


def test_ocr_multipart_form_values_are_json_decoded_before_forwarding(gateway: Gateway) -> None:
    """A multipart upload carries `pages=[0,2]` and `image_limit=3` as text fields. Mistral must receive `pages` as
    the JSON list `[0, 2]` and `image_limit` as the integer 3, because the endpoint `json.loads` every form value
    it can parse."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.client.post(
            _OCR,
            data={"model": model, "pages": "[0,2]", "image_limit": "3"},
            files={"file": ("doc.pdf", PDF_BYTES + uuid.uuid4().bytes, "application/pdf")},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    body: Final = json_body(request)
    assert (body["pages"], body["image_limit"]) == ([0, 2], 3)


# ---------------------------------------------------------------- Mistral wire


def test_mistral_ocr_forwards_supported_params_and_drops_unknown_ones(gateway: Gateway) -> None:
    """The caller sends `pages`, `include_image_base64`, `table_format`, `extract_header` and an unknown field
    `foo_bar`. Mistral must get exactly `model`, `document` and the four supported params with the same values,
    and no `foo_bar` (MistralOcrConfig.map_ocr_params selects only the supported list)."""
    document: Final = _document()
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.request(
            "POST",
            _OCR,
            {
                "model": model,
                "document": document,
                "pages": [0],
                "include_image_base64": False,
                "table_format": "html",
                "extract_header": True,
                "foo_bar": "dropped",
            },
        )
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", "/v1/ocr")
    assert request.headers["authorization"] == f"Bearer {MISTRAL_KEY}"
    assert json_body(request) == {
        "model": "mistral-ocr-latest",
        "document": document,
        "pages": [0],
        "include_image_base64": False,
        "table_format": "html",
        "extract_header": True,
    }


def test_mistral_ocr_api_base_ending_in_v1_is_not_doubled(gateway: Gateway) -> None:
    """The deployment's `api_base` is `<peer>/v1`. Mistral must get the POST at `/v1/ocr`, not `/v1/v1/ocr`,
    because the Rust URL builder completes only the missing suffix (`ApiUrl::complete_path(["v1", "ocr"])`).
    UNVERIFIED: complete_path semantics read from its name and the Azure unit test, not its body."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest", api_base=f"{wire.url}/v1", api_key=MISTRAL_KEY
        )
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _document()})
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert path_of(request) == "/v1/ocr"


def test_mistral_ocr_response_carries_pages_usage_and_document_annotation(gateway: Gateway) -> None:
    """Mistral answers two pages with dimensions, `usage_info` `{"pages_processed": 2, "doc_size_bytes": 2048}` and
    a `document_annotation` string. The caller must get `object` `ocr`, both pages with `index`, `markdown`,
    `images` and `dimensions` unchanged, `usage_info.pages_processed` 2 and the same `document_annotation`."""
    with (
        wire_server(lambda _r: mistral_ocr_reply(2, document_annotation='{"title": "t"}')) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _document()})
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert len(wire.drain()) == 1
    assert body["object"] == "ocr"
    assert body["pages"] == [
        {"index": 0, "markdown": "# page 0", "images": [], "dimensions": {"dpi": 200, "height": 1100, "width": 850}},
        {"index": 1, "markdown": "# page 1", "images": [], "dimensions": {"dpi": 200, "height": 1100, "width": 850}},
    ]
    assert body["usage_info"]["pages_processed"] == 2  # type: ignore[index]
    assert body["document_annotation"] == '{"title": "t"}'


def test_mistral_ocr_error_status_reaches_the_caller(gateway: Gateway) -> None:
    """Mistral answers 422 `{"object": "error", "message": "Document could not be fetched", "type":
    "invalid_request_error"}`. The caller must get 422 and the provider message.
    UNVERIFIED: whether 422 is kept or folded into 400 by the Rust error mapping."""
    with (
        wire_server(
            lambda _r: json_reply(
                {"object": "error", "message": "Document could not be fetched", "type": "invalid_request_error"}, 422
            )
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _document()})
        assert response.status_code == 422, response.text
        assert "Document could not be fetched" in response.text
        assert len(wire.drain()) == 1


def test_mistral_ocr_reply_with_a_null_model_is_refused_instead_of_reported(gateway: Gateway) -> None:
    """Mistral answers 200 with `"model": null`. The Rust normalizer refuses an explicit null model
    (`ResponseField { path: "model" }`), so the caller must get an error status and no 200 with a null model.
    UNVERIFIED: the exact status code; 500 or 502 are both plausible."""
    with (
        wire_server(lambda _r: Reply(body=json.dumps({"model": None, "pages": []}).encode())) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _document()})
        assert response.status_code >= 500, response.text
        assert len(wire.drain()) == 1


# ---------------------------------------------------------------- Azure AI


def test_azure_ai_ocr_posts_the_mistral_body_to_the_azure_mistral_path_with_a_bearer_key(gateway: Gateway) -> None:
    """An `azure_ai/mistral-document-ai-2505` deployment receives an inline PDF data URL. Azure must get one POST to
    `/providers/mistral/azure/ocr` with `Authorization: Bearer <key>` and the Mistral body `{"model":
    "mistral-document-ai-2505", "document": <same document>}`."""
    document: Final = _inline_document()
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure_ai/mistral-document-ai-2505", api_base=wire.url, api_key=AZURE_KEY)
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": document})
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert (request.method, path_of(request)) == ("POST", "/providers/mistral/azure/ocr")
    assert request.headers["authorization"] == f"Bearer {AZURE_KEY}"
    assert json_body(request) == {"model": "mistral-document-ai-2505", "document": document}


def test_azure_ai_ocr_keeps_the_api_base_query_and_does_not_repeat_a_complete_path(gateway: Gateway) -> None:
    """The deployment's `api_base` is already `<peer>/providers/mistral/azure/ocr?tenant=a`. Azure must get the POST
    at `/providers/mistral/azure/ocr` once, with `tenant=a` still in the query (AzureAiOcrConfig.build_ocr_url,
    whose unit test pins both cases)."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure_ai/mistral-document-ai-2505",
            api_base=f"{wire.url}/providers/mistral/azure/ocr?tenant=a",
            api_key=AZURE_KEY,
        )
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _inline_document()})
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert path_of(request) == "/providers/mistral/azure/ocr"
    assert query_of(request) == {"tenant": ["a"]}


def test_azure_ai_ocr_downloads_a_remote_document_and_sends_it_inline(gateway: Gateway) -> None:
    """The caller names `document_url` `<peer>/files/doc.pdf`, which the peer serves as `application/pdf`. Azure AI
    accepts only inline documents, so the proxy must GET the file first and then POST
    `/providers/mistral/azure/ocr` with `document_url` `data:application/pdf;base64,<bytes>`.
    UNVERIFIED: the OCR media fetcher may refuse a loopback URL as an SSRF guard, in which case this needs a
    public-looking host in front of the peer."""
    pdf: Final = PDF_BYTES + uuid.uuid4().bytes

    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=pdf, content_type="application/pdf")
        return mistral_ocr_reply(1)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure_ai/mistral-document-ai-2505", api_base=wire.url, api_key=AZURE_KEY)
        response: Final = gateway.request(
            "POST",
            _OCR,
            {"model": model, "document": {"type": "document_url", "document_url": f"{wire.url}/files/doc.pdf"}},
        )
        assert response.status_code == 200, response.text
        fetch, ocr_call = wire.drain()
    assert (fetch.method, path_of(fetch)) == ("GET", "/files/doc.pdf")
    assert "authorization" not in fetch.headers
    assert json_body(ocr_call)["document"] == {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64.b64encode(pdf).decode()}",
    }


def test_azure_ai_ocr_bills_pages_at_the_deployment_page_rate(gateway: Gateway) -> None:
    """An Azure AI OCR deployment priced at `ocr_cost_per_page` 0.002 processes 3 pages. The response cost header,
    the spend row (`call_type` `aocr`, `custom_llm_provider` `azure_ai`) and the key spend must all be 3 x 0.002."""
    with wire_server(lambda _r: mistral_ocr_reply(3)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="azure_ai/mistral-document-ai-2505",
            api_base=wire.url,
            api_key=AZURE_KEY,
            ocr_cost_per_page=_PAGE_RATE,
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _inline_document()}, key=key)
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(3 * _PAGE_RATE)
        (row,) = logged_rows(key)
        assert (row["call_type"], row["custom_llm_provider"], row["status"]) == ("aocr", "azure_ai", "success")
        assert float(str(row["spend"])) == pytest.approx(3 * _PAGE_RATE)
        settled_key_spend(key, 3 * _PAGE_RATE)
        assert len(wire.drain()) == 1


# ---------------------------------------------------------------- page pricing rules


def test_ocr_credit_pricing_wins_over_page_pricing_when_credits_are_reported(gateway: Gateway) -> None:
    """A Mistral deployment sets both `ocr_cost_per_page` 0.002 and `ocr_cost_per_credit` 0.01, and Mistral reports
    `pages_processed` 4 with `credits` 1.5. The cost must be 1.5 x 0.01, not 4 x 0.002, because credits are
    checked first in `ocr_cost`.
    UNVERIFIED: assumes `ocr_cost_per_credit` is accepted as a deployment price like `ocr_cost_per_page`."""
    usage: Final = {"pages_processed": 4, "credits": 1.5, "doc_size_bytes": 4096}
    with wire_server(lambda _r: mistral_ocr_reply(4, usage=usage)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest",
            api_base=wire.url,
            api_key=MISTRAL_KEY,
            ocr_cost_per_page=_PAGE_RATE,
            ocr_cost_per_credit=0.01,
        )
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _document()})
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(1.5 * 0.01)
        assert len(wire.drain()) == 1


def test_ocr_annotation_pages_fall_back_to_the_page_rate_without_an_annotation_price(gateway: Gateway) -> None:
    """A Mistral deployment sets only `ocr_cost_per_page` 0.002 and Mistral reports 3 pages processed and 2 pages
    processed for annotation. The cost must be 3 x 0.002 + 2 x 0.002, because the annotation rate defaults to the
    page rate when no `annotation_cost_per_page` is set."""
    usage: Final = {"pages_processed": 3, "pages_processed_annotation": 2, "doc_size_bytes": 4096}
    with wire_server(lambda _r: mistral_ocr_reply(3, usage=usage)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY, ocr_cost_per_page=_PAGE_RATE
        )
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _document()})
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(5 * _PAGE_RATE)
        assert len(wire.drain()) == 1


def test_ocr_without_usage_pages_on_a_page_priced_model_bills_nothing_but_still_answers(gateway: Gateway) -> None:
    """Mistral answers two pages but `usage_info` has `credits` 2 and no `pages_processed`, and the deployment has
    only `ocr_cost_per_page`. The caller must still get 200 with the pages; the cost raises inside `ocr_cost`
    (`OCR response pages_processed is None`), which the proxy logs instead of failing the request.
    UNVERIFIED: whether the cost header is absent or `0.0`; the test only pins that the request succeeds."""
    usage: Final = {"credits": 2, "doc_size_bytes": 4096}
    with wire_server(lambda _r: mistral_ocr_reply(2, usage=usage)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY, ocr_cost_per_page=_PAGE_RATE
        )
        response: Final = gateway.request("POST", _OCR, {"model": model, "document": _document()})
        assert response.status_code == 200, response.text
        assert len(JSON_OBJECT.validate_json(response.content)["pages"]) == 2  # type: ignore[arg-type]
        assert len(wire.drain()) == 1


def test_ocr_alias_route_without_v1_reaches_the_same_upstream(gateway: Gateway) -> None:
    """The proxy serves OCR at both `/v1/ocr` and `/ocr`. A call to `/ocr` must produce the same single POST to
    Mistral `/v1/ocr` and a 200, so clients configured without the version prefix keep working."""
    with wire_server(_two_pages) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=wire.url, api_key=MISTRAL_KEY)
        response: Final = gateway.request("POST", "/ocr", {"model": model, "document": _document()})
        assert response.status_code == 200, response.text
        (request,) = wire.drain()
    assert path_of(request) == "/v1/ocr"
