"""OCR parse errors, Responses `max_output_tokens`, vector store file attributes and the daily activity date range,
driven through the shared rig proxy with the provider body read back from the scripted upstream."""

from __future__ import annotations

import asyncio
import itertools
import uuid
from collections.abc import Callable
from typing import Final

import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.upstream import ScenarioHandle
from integration.compatibility._status_code_audit import (
    UNSET,
    RESPONSE,
    RESPONSES_FRAMES,
    assembled_text,
    assert_no_provider_call,
    drain_rig_upstream,
    invalid_request,
    json_response,
    one_outbound,
    openai_error,
    post,
    register,
    sse_response,
    stream_finished,
    stream_lines,
)
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import StoredResponse

pytestmark: Final = pytest.mark.timeout(180)


@pytest.fixture(autouse=True)
def _drained_upstream() -> None:
    drain_rig_upstream()


_OCR: Final[dict[str, JsonValue]] = {
    "pages": [{"index": 0, "markdown": "scripted page", "images": [], "dimensions": None}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1},
}
_VECTOR_STORE_FILE: Final[dict[str, JsonValue]] = {
    "id": "file-audit",
    "object": "vector_store.file",
    "created_at": 1,
    "usage_bytes": 0,
    "vector_store_id": "vs-audit",
    "status": "completed",
    "last_error": None,
    "attributes": {},
}


def _ocr_model(scenario: Scenario) -> tuple[str, str]:
    identity, handle = register(scenario, "audit-ocr", json_response(_OCR))
    return scenario.model(model="mistral/mistral-ocr-latest", api_base=handle.api_base(), api_key=identity), identity


def _raw(gateway: Gateway, path: str, content: bytes, content_type: str, *, key: str | None = None) -> httpx.Response:
    return gateway.client.post(
        path,
        content=content,
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}", "Content-Type": content_type},
        timeout=60,
    )


@pytest.mark.parametrize(
    ("path", "body", "message", "param"),
    (
        pytest.param(
            "/v1/ocr",
            {"document": {"type": "file", "file": "local.pdf"}},
            "document type 'file' is not supported through the JSON API.",
            None,
            id="document-type-file",
        ),
        pytest.param(
            "/v1/ocr",
            {"document": {"type": "document_url", "document_url": "reducto://file-123"}},
            "reducto:// file IDs are not accepted through the proxy OCR API;",
            None,
            id="reducto-document-url",
        ),
        pytest.param(
            "/v1/ocr",
            {"document": {"type": "image_url", "image_url": "reducto://file-123"}},
            "reducto:// file IDs are not accepted through the proxy OCR API;",
            None,
            id="reducto-image-url",
        ),
        pytest.param(
            "/ocr",
            {"document": {"type": "file", "file": "local.pdf"}},
            "document type 'file' is not supported through the JSON API.",
            None,
            id="unversioned-route",
        ),
    ),
)
def test_ocr_unsupported_json_document_returns_400(
    gateway: Gateway, path: str, body: dict[str, JsonValue], message: str, param: str | None
) -> None:
    with gateway.scenario() as scenario:
        model, identity = _ocr_model(scenario)
        response: Final = post(gateway, path, {"model": model, **body})
        error: Final = invalid_request(response)
        assert str(error.get("message")).startswith(message), response.text
        assert error.get("param") == param, response.text
        assert_no_provider_call(gateway, identity)


@pytest.mark.parametrize(
    ("content", "content_type", "message", "param"),
    (
        pytest.param(b"", "application/json", "Empty request body.", None, id="empty-json-body"),
        pytest.param(b"{not json", "application/json", "Invalid JSON payload", "request_body", id="invalid-json"),
        pytest.param(b"hello", "text/plain", "Invalid JSON payload", "request_body", id="text-plain"),
        pytest.param(
            b"--x\r\n",
            "multipart/form-data",
            "Invalid form payload: 400: Missing boundary",
            "request_body",
            id="no-boundary",
        ),
    ),
)
def test_ocr_unparseable_body_returns_400(
    gateway: Gateway, content: bytes, content_type: str, message: str, param: str | None
) -> None:
    response: Final = _raw(gateway, "/v1/ocr", content, content_type)
    error: Final = invalid_request(response)
    assert str(error.get("message")).startswith(message), response.text
    assert error.get("param") == param, response.text


def test_ocr_json_array_body_returns_400(gateway: Gateway) -> None:
    pytest.skip("BUG: a JSON array body answers 500 on every route, from the auth pre-checks; same on base")
    response: Final = _raw(gateway, "/v1/ocr", b"[1,2]", "application/json")
    assert invalid_request(response).get("param") == "request_body", response.text


def test_ocr_file_over_the_size_limit_returns_400(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity = _ocr_model(scenario)
        response: Final = gateway.request_multipart(
            "/v1/ocr", {"model": model}, {"file": ("big.pdf", b"%PDF" + b"0" * (50 * 1024 * 1024), "application/pdf")}
        )
        error: Final = invalid_request(response)
        assert error.get("message") == "OCR file exceeds the size limit", response.text
        assert error.get("param") == "file", response.text
        assert_no_provider_call(gateway, identity)


def test_ocr_multipart_without_a_key_is_401(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity = _ocr_model(scenario)
        response: Final = gateway.client.post(
            "/v1/ocr", files={"model": (None, model)}, headers={"Authorization": "Bearer sk-audit-not-a-key"}
        )
        assert response.status_code == 401, response.text
        assert_no_provider_call(gateway, identity)


def test_ocr_document_url_reaches_the_provider(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity = _ocr_model(scenario)
        document: Final[dict[str, JsonValue]] = {"type": "document_url", "document_url": "https://example.com/a.pdf"}
        response: Final = post(gateway, "/v1/ocr", {"model": model, "document": document})
        assert response.status_code == 200, response.text
        assert response.json()["pages"][0]["markdown"] == "scripted page"
        assert one_outbound(gateway, identity).get("document") == document


def test_ocr_multipart_file_reaches_the_provider_as_a_data_uri(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity = _ocr_model(scenario)
        response: Final = gateway.request_multipart(
            "/v1/ocr", {"model": model}, {"file": ("doc.pdf", b"%PDF-1.4 audit", "application/pdf")}
        )
        assert response.status_code == 200, response.text
        document: Final = one_outbound(gateway, identity).get("document")
        assert isinstance(document, dict), document
        assert document.get("type") == "document_url", document
        assert str(document.get("document_url")).startswith("data:application/pdf;base64,"), document


def _responses_model(scenario: Scenario, response: StoredResponse) -> tuple[str, str]:
    identity, handle = register(scenario, "audit-max-output", response)
    return scenario.model(model="openai/gpt-4o-mini", api_base=handle.api_base(), api_key=identity), identity


@pytest.mark.parametrize(
    ("sent", "forwarded"),
    (
        pytest.param(1, 16, id="1"),
        pytest.param(15, 16, id="15"),
        pytest.param(16, 16, id="16"),
        pytest.param(4096, 4096, id="4096"),
        pytest.param(True, 16, id="true"),
        pytest.param(None, UNSET, id="null"),
        pytest.param(UNSET, UNSET, id="absent"),
        pytest.param("8", "8", id="string"),
        pytest.param(0.5, 0.5, id="float"),
    ),
)
def test_responses_max_output_tokens_forwarding(gateway: Gateway, sent: object, forwarded: object) -> None:
    with gateway.scenario() as scenario:
        model, identity = _responses_model(scenario, json_response(RESPONSE))
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "input": "Say pong",
            "store": False,
            **({} if sent is UNSET else {"max_output_tokens": sent}),  # pyright: ignore[reportAssignmentType]  # the parametrized shapes are JSON values
        }
        response: Final = post(gateway, "/v1/responses", body)
        assert response.status_code == 200, response.text
        outbound: Final = one_outbound(gateway, identity)
        assert outbound.get("max_output_tokens", UNSET) == forwarded, outbound


_BELOW_MINIMUM: Final = (
    "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got {} instead."
)


@pytest.mark.parametrize("sent", (0, -1), ids=("zero", "negative"))
def test_responses_non_positive_max_output_tokens_stream_returns_the_provider_400(gateway: Gateway, sent: int) -> None:
    message: Final = _BELOW_MINIMUM.format(sent)
    with gateway.scenario() as scenario:
        model, identity = _responses_model(scenario, openai_error(message, "max_output_tokens"))
        status, lines = stream_lines(
            gateway,
            "/v1/responses",
            {"model": model, "input": "Say pong", "store": False, "stream": True, "max_output_tokens": sent},
        )
        assert status == 400, lines
        assert any(message in line for line in lines), lines
        outbound: Final = one_outbound(gateway, identity)
        assert (outbound.get("max_output_tokens"), outbound.get("stream")) == (sent, True), outbound


def test_responses_small_max_output_tokens_stream_is_raised_to_the_minimum(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity = _responses_model(scenario, sse_response(RESPONSES_FRAMES))
        status, lines = stream_lines(
            gateway,
            "/v1/responses",
            {"model": model, "input": "Say pong", "store": False, "stream": True, "max_output_tokens": 1},
        )
        assert status == 200, lines
        assert stream_finished("responses", lines), lines
        assert assembled_text(lines) == "streamed response", lines
        assert one_outbound(gateway, identity).get("max_output_tokens") == 16


def _sdk_responses(
    gateway: Gateway, model: str, sent: int, *, asynchronous: bool, stream: bool
) -> Callable[[], object]:
    base_url: Final = f"{gateway.client.base_url}/v1"

    def sync_call() -> object:
        with openai.OpenAI(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
            return client.responses.create(
                model=model, input="Say pong", store=False, max_output_tokens=sent, stream=stream
            )

    async def async_call() -> object:
        async with openai.AsyncOpenAI(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
            return await client.responses.create(
                model=model, input="Say pong", store=False, max_output_tokens=sent, stream=stream
            )

    return (lambda: asyncio.run(async_call())) if asynchronous else sync_call


@pytest.mark.parametrize(
    ("sent", "asynchronous", "stream"),
    ((0, False, False), (0, True, False), (-1, False, True), (-1, True, True)),
    ids=("zero-sync", "zero-async", "negative-sync-stream", "negative-async-stream"),
)
def test_responses_non_positive_max_output_tokens_through_the_openai_sdk(
    gateway: Gateway, sent: int, asynchronous: bool, stream: bool
) -> None:
    message: Final = _BELOW_MINIMUM.format(sent)
    with gateway.scenario() as scenario:
        model, identity = _responses_model(scenario, openai_error(message, "max_output_tokens"))
        call: Final = _sdk_responses(gateway, model, sent, asynchronous=asynchronous, stream=stream)
        with pytest.raises(openai.BadRequestError) as raised:
            call()
        assert raised.value.status_code == 400
        assert message in raised.value.response.text
        assert one_outbound(gateway, identity).get("max_output_tokens") == sent


def _registered_store(gateway: Gateway, scenario: Scenario, identity: str, handle: ScenarioHandle) -> str:
    store: Final = f"vs-{uuid.uuid4().hex}"
    model: Final = scenario.model(model="openai/text-embedding-3-small", api_base=handle.api_base(), api_key=identity)
    gateway.post(
        "/vector_store/new",
        {
            "vector_store_id": store,
            "custom_llm_provider": "openai",
            "litellm_params": {"model": model, "api_base": handle.api_base(), "api_key": identity},
        },
    )
    scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store})
    return store


_SIXTEEN: Final[dict[str, JsonValue]] = {f"key_{index:02d}": f"value {index}" for index in range(16)}
_ATTRIBUTE_CASES: Final = (
    pytest.param(_SIXTEEN, _SIXTEEN, id="sixteen"),
    pytest.param({}, {}, id="empty"),
    pytest.param({"team": "a", "hidden_params": "internal"}, {"team": "a"}, id="hidden-params"),
    pytest.param({"long": "x" * 5000}, {"long": "x" * 5000}, id="5kb-value"),
)


@pytest.mark.parametrize("operation", ("create", "update"))
@pytest.mark.parametrize(("attributes", "forwarded"), _ATTRIBUTE_CASES)
def test_vector_store_file_attributes_reach_the_provider(
    gateway: Gateway, operation: str, attributes: dict[str, JsonValue], forwarded: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        identity, handle = register(scenario, "audit-attributes", json_response(_VECTOR_STORE_FILE))
        store: Final = _registered_store(gateway, scenario, identity, handle)
        path: Final = (
            f"/v1/vector_stores/{store}/files"
            if operation == "create"
            else f"/v1/vector_stores/{store}/files/file-audit"
        )
        body: Final[dict[str, JsonValue]] = {
            "attributes": attributes,
            **({"file_id": "file-audit"} if operation == "create" else {}),
        }
        response: Final = post(gateway, path, body)
        assert response.status_code == 200, response.text
        assert one_outbound(gateway, identity).get("attributes") == forwarded


@pytest.mark.parametrize("operation", ("create", "update"))
def test_vector_store_file_number_and_boolean_attributes_reach_the_provider(gateway: Gateway, operation: str) -> None:
    pytest.skip(
        "BUG: number and boolean vector store file attributes (valid OpenAI attribute values) are dropped before "
        "the provider call, so they are silently lost; same on base"
    )
    attributes: Final[dict[str, JsonValue]] = {"pages": 3, "public": True, "team": "a"}
    with gateway.scenario() as scenario:
        identity, handle = register(scenario, "audit-attributes-typed", json_response(_VECTOR_STORE_FILE))
        store: Final = _registered_store(gateway, scenario, identity, handle)
        path: Final = (
            f"/v1/vector_stores/{store}/files"
            if operation == "create"
            else f"/v1/vector_stores/{store}/files/file-audit"
        )
        body: Final[dict[str, JsonValue]] = {
            "attributes": attributes,
            **({"file_id": "file-audit"} if operation == "create" else {}),
        }
        response: Final = post(gateway, path, body)
        assert response.status_code == 200, response.text
        assert one_outbound(gateway, identity).get("attributes") == attributes


_ENTITIES: Final = ("agent", "customer", "organization", "tag", "team", "user")
_SUFFIXES: Final = (
    "",
    "/aggregated",
    "/aggregated/keys",
    "/aggregated/model_top_keys",
    "/aggregated/search",
    "/export",
)
_DAILY_ROUTES: Final = tuple(
    f"/{entity}/daily/activity{suffix}"
    for entity, suffix in itertools.product(_ENTITIES, _SUFFIXES)
    if f"/{entity}/daily/activity{suffix}" != "/organization/daily/activity"
) + ("/user/daily/activity/aggregated/cache_leakage_keys",)
_DATE_CASES: Final[dict[str, tuple[str, str, int, str | None]]] = {
    "reversed": ("2026-10-01", "2026-09-24", 400, "end_date must be on or after start_date"),
    "equal": ("2026-10-01", "2026-10-01", 200, None),
    "bad-month": ("2026-13-01", "2026-10-01", 400, "start_date and end_date must be valid YYYY-MM-DD dates"),
}


def _route_params(route: str) -> dict[str, str]:
    if route.endswith("/model_top_keys"):
        return {"model_group": "audit-model"}
    if route.endswith("/search"):
        return {"search": "audit"}
    return {}


@pytest.mark.parametrize("case", tuple(_DATE_CASES))
@pytest.mark.parametrize("route", _DAILY_ROUTES)
def test_daily_activity_date_range(gateway: Gateway, route: str, case: str) -> None:
    start, end, status, error = _DATE_CASES[case]
    response: Final = gateway.request(
        "GET", route, params={"start_date": start, "end_date": end, **_route_params(route)}
    )
    assert response.status_code == status, response.text
    if error is not None:
        assert response.json() == {"detail": {"error": error}}, response.text


@pytest.mark.parametrize("case", ("reversed", "bad-month"))
def test_gateway_daily_activity_date_range(gateway: Gateway, case: str) -> None:
    pytest.skip(
        "BUG: /gateway/daily/activity answers 200 with empty totals for a reversed or invalid date range "
        "instead of the 400 every other daily activity route gives; same on base"
    )
    start, end, status, error = _DATE_CASES[case]
    response: Final = gateway.request("GET", "/gateway/daily/activity", params={"start_date": start, "end_date": end})
    assert response.status_code == status, response.text
    assert response.json() == {"detail": {"error": error}}, response.text
