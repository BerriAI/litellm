from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.extraction import ExtractionResponse

_CLAIRIC_URL: Final = "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clairic"
_STATE: Final = "Invoice #4471 from Acme Tools, dated 2026-03-03. 12 hammers. Total $174. Payment due in 30 days."
_FIELDS: Final[Mapping[str, Mapping[str, object]]] = MappingProxyType(
    {
        "vendor": {"type": "text", "instructions": "Who sent the invoice?"},
        "invoice_date": {"type": "date"},
        "total": {"type": "number", "unit": "USD"},
        "items": {"type": "list", "items": "text"},
        "days_to_pay": {"type": "estimate", "min": 0, "max": 90, "provider_field": "kept"},
    }
)
_INPUT_TOKENS: Final = 416
_RESULT: Final[Mapping[str, object]] = {
    "model": "clairic",
    "answers": {
        "vendor": {
            "type": "text",
            "p_null": 0.0177,
            "span": [19, 29],
            "guess": "Acme Tools",
            "confidence": 0.9998,
            "value": "Acme Tools",
        },
        "invoice_date": {"type": "date", "p_null": 6.4e-07, "confidence": 0.99, "value": "2026-03-03"},
        "total": {"type": "number", "p_null": 4.2e-07, "guess": 174, "confidence": 0.99, "value": 174},
        "items": {"type": "list", "value": ["12", "hammers"], "spans": [[51, 53], [54, 61]], "confidence": 0.07},
        "days_to_pay": {
            "type": "estimate",
            "value": 31,
            "mean": 30.76,
            "median": 30,
            "p10": 29,
            "p90": 34,
            "p_below": 1.2e-06,
            "p_above": 1.2e-06,
        },
    },
    "truncated": False,
    "usage": {
        "state_tokens": 52,
        "image_tokens": 0,
        "schema_tokens": 312,
        "input_tokens": _INPUT_TOKENS,
        "output_tokens": 0,
        "passes": 2,
    },
}
_ENVELOPE: Final[Mapping[str, object]] = {"result": _RESULT, "success": True, "errors": [], "messages": []}
_PNG_DATA_URL: Final = "data:image/png;base64,AA=="


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)


def _assert_invoice_answers(response: ExtractionResponse) -> None:
    assert response.answers["vendor"].value == "Acme Tools"
    assert response.answers["vendor"].span == (19, 29)
    assert response.answers["items"].spans == ((51, 53), (54, 61))
    assert response.answers["days_to_pay"].median == 30
    assert response.usage.input_tokens == _INPUT_TOKENS
    assert response.usage.schema_tokens == 312
    assert response.truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("cloudflare/clairic", "cloudflare/@cf/cloudflare/clairic"))
@pytest.mark.usefixtures("cloudflare_env")
async def test_clairic_forwards_the_caller_body_and_unwraps_the_envelope(
    respx_mock: respx.MockRouter, model: str
) -> None:
    route: Final = respx_mock.post(_CLAIRIC_URL).respond(json=_ENVELOPE)

    response: Final = await litellm.aextraction(
        model=model,
        state=_STATE,
        fields=_FIELDS,
        images=[_PNG_DATA_URL, {"content_type": "image/jpeg", "base64": "AA=="}],
        reference_date="2026-03-04",
        options={"include_quantiles": True},
    )

    assert route.call_count == 1
    request: Final = route.calls[0].request
    assert request.headers["authorization"] == "Bearer cloudflare-key"
    assert json.loads(request.content) == {
        "model": "clairic",
        "state": _STATE,
        "fields": _FIELDS,
        "images": [_PNG_DATA_URL, {"content_type": "image/jpeg", "base64": "AA=="}],
        "reference_date": "2026-03-04",
        "options": {"include_quantiles": True},
    }
    _assert_invoice_answers(response)
    assert response.model == model
    assert response.hidden_params["model"] == "cloudflare/@cf/cloudflare/clairic"
    assert response.model_dump(mode="json", exclude={"model"}) == {
        key: value for key, value in _RESULT.items() if key != "model"
    }


@pytest.mark.usefixtures("cloudflare_env")
def test_sync_extraction_reads_an_unwrapped_result(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_CLAIRIC_URL).respond(json=_RESULT)

    response: Final = litellm.extraction(model="cloudflare/clairic", state=_STATE, fields=_FIELDS)

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {"model": "clairic", "state": _STATE, "fields": _FIELDS}
    _assert_invoice_answers(response)


def _clairic_transport() -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == _CLAIRIC_URL
        return httpx.Response(200, json=_ENVELOPE)

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_env")
async def test_async_extraction_sends_through_the_callers_client(respx_mock: respx.MockRouter) -> None:
    response: Final = await litellm.aextraction(
        model="cloudflare/clairic",
        state=_STATE,
        fields=_FIELDS,
        client=AsyncHTTPHandler(transport=_clairic_transport()),
    )

    assert len(respx_mock.calls) == 0
    _assert_invoice_answers(response)


@pytest.mark.usefixtures("cloudflare_env")
def test_sync_extraction_sends_through_the_callers_client(respx_mock: respx.MockRouter) -> None:
    response: Final = litellm.extraction(
        model="cloudflare/clairic",
        state=_STATE,
        fields=_FIELDS,
        client=HTTPHandler(client=httpx.Client(transport=_clairic_transport())),
    )

    assert len(respx_mock.calls) == 0
    _assert_invoice_answers(response)


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_env")
async def test_a_cloudflare_bad_input_is_a_bad_request_with_the_cloudflare_message(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(_CLAIRIC_URL).respond(
        status_code=400,
        json={
            "errors": [{"message": "AiError: Bad input: fields.total.decimals must be 0 to 6", "code": 5006}],
            "success": False,
            "result": {},
            "messages": [],
        },
    )

    with pytest.raises(litellm.BadRequestError, match=re.escape("fields.total.decimals must be 0 to 6")):
        await litellm.aextraction(
            model="cloudflare/clairic", state=_STATE, fields={"total": {"type": "number", "decimals": 9}}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "images"),
    (
        ({}, None),
        ({"total": {"type": "money"}}, None),
        ({"bad name": {"type": "text"}}, None),
        ({"total": {"type": "number"}}, ["https://example.com/invoice.png"]),
        ({"total": {"type": "number"}}, [_PNG_DATA_URL] * 5),
    ),
    ids=("no_fields", "unknown_type", "invalid_field_name", "remote_image", "too_many_images"),
)
@pytest.mark.usefixtures("cloudflare_env")
async def test_an_invalid_request_is_refused_before_http(
    respx_mock: respx.MockRouter, fields: Mapping[str, object], images: list[str] | None
) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Extraction request"):
        await litellm.aextraction(model="cloudflare/clairic", state=_STATE, fields=fields, images=images)

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("unset", "message"),
    (
        ("CLOUDFLARE_ACCOUNT_ID", "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID"),
        ("CLOUDFLARE_API_KEY", "Missing API key for Extraction provider 'cloudflare'"),
    ),
)
@pytest.mark.usefixtures("cloudflare_env")
async def test_missing_cloudflare_credentials_are_refused_before_http(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, unset: str, message: str
) -> None:
    monkeypatch.delenv(unset)

    with pytest.raises(litellm.BadRequestError, match=message):
        await litellm.aextraction(model="cloudflare/clairic", state=_STATE, fields=_FIELDS)

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_a_provider_without_extraction_is_refused(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match=r"Unknown Extraction provider 'openai'\. Supported.*cloudflare"):
        await litellm.aextraction(model="openai/gpt-5.5", state=_STATE, fields=_FIELDS, api_key="k")

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("cloudflare/clairic", "cloudflare/@cf/cloudflare/clairic"))
@pytest.mark.usefixtures("cloudflare_env")
async def test_clairic_cost_bills_input_tokens_at_the_cost_map_price(respx_mock: respx.MockRouter, model: str) -> None:
    respx_mock.post(_CLAIRIC_URL).respond(json=_ENVELOPE)

    response: Final = await litellm.aextraction(model=model, state=_STATE, fields=_FIELDS)

    expected_cost: Final = _INPUT_TOKENS * float(litellm.model_cost[model]["input_cost_per_token"])
    assert expected_cost > 0
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected_cost)
    assert response.hidden_params["response_cost"] == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_router_extraction_sends_the_deployment_model_and_credentials(respx_mock: respx.MockRouter) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "invoice-reader",
                "litellm_params": {
                    "model": "cloudflare/clairic",
                    "api_key": "deployment-key",
                    "api_base": "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1",
                    "additional_drop_params": ["temperature"],
                },
            }
        ]
    )
    route: Final = respx_mock.post(_CLAIRIC_URL).respond(json=_ENVELOPE)

    response: Final = await router.aextraction(model="invoice-reader", state=_STATE, fields=_FIELDS)

    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == "Bearer deployment-key"
    assert json.loads(route.calls[0].request.content) == {"model": "clairic", "state": _STATE, "fields": _FIELDS}
    assert response.model == "cloudflare/clairic"
    _assert_invoice_answers(response)


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_env")
async def test_litellm_params_stay_local(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_CLAIRIC_URL).respond(json=_ENVELOPE)

    await litellm.aextraction(
        model="cloudflare/clairic",
        state=_STATE,
        fields=_FIELDS,
        metadata={"team": "billing"},
        num_retries=0,
        stream=False,
        additional_drop_params=["language"],
    )

    assert json.loads(route.calls[0].request.content) == {"model": "clairic", "state": _STATE, "fields": _FIELDS}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("language", "priority", "session_id", "mock_delay", "extra_body"))
@pytest.mark.usefixtures("cloudflare_env")
async def test_a_key_clairic_does_not_take_is_refused_before_http_without_drop_params(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(litellm.UnsupportedParamsError, match=re.escape(f"does not support parameters: ['{name}']")):
        await litellm.aextraction(model="cloudflare/clairic", state=_STATE, fields=_FIELDS, **{name: "x"})

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_env")
async def test_additional_drop_params_drops_only_the_keys_it_names(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    route: Final = respx_mock.post(_CLAIRIC_URL).respond(json=_ENVELOPE)

    await litellm.aextraction(
        model="cloudflare/clairic",
        state=_STATE,
        fields=_FIELDS,
        temperature=0.2,
        language="en",
        additional_drop_params=["temperature", "language"],
    )
    with pytest.raises(litellm.UnsupportedParamsError, match=re.escape("does not support parameters: ['seed']")):
        await litellm.aextraction(
            model="cloudflare/clairic", state=_STATE, fields=_FIELDS, seed=7, additional_drop_params=["temperature"]
        )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {"model": "clairic", "state": _STATE, "fields": _FIELDS}


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.standard_logging_object: Mapping[str, object] | None = None
        self.stream: object = None

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.stream = kwargs.get("stream")
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            self.standard_logging_object = standard_logging_object


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_env")
async def test_stream_is_refused_before_http_without_drop_params(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match=re.escape("does not support parameters: ['stream']")):
        await litellm.aextraction(model="cloudflare/clairic", state=_STATE, fields=_FIELDS, stream=True)

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "value"),
    (("temperature", 0.2), ("user", "end-user-1"), ("seed", 7), ("response_format", {"type": "json_object"})),
)
@pytest.mark.usefixtures("cloudflare_env")
async def test_openai_generation_params_are_refused_before_http_without_drop_params(
    respx_mock: respx.MockRouter, name: str, value: object
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match=re.escape(f"does not support parameters: ['{name}']")):
        await litellm.aextraction(model="cloudflare/clairic", state=_STATE, fields=_FIELDS, **{name: value})

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_env")
async def test_openai_generation_params_are_dropped_with_drop_params(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_CLAIRIC_URL).respond(json=_ENVELOPE)

    await litellm.aextraction(
        model="cloudflare/clairic",
        state=_STATE,
        fields=_FIELDS,
        temperature=0.2,
        user="end-user-1",
        seed=7,
        drop_params=True,
    )

    assert json.loads(route.calls[0].request.content) == {"model": "clairic", "state": _STATE, "fields": _FIELDS}


@pytest.mark.asyncio
@pytest.mark.usefixtures("cloudflare_env")
async def test_a_dropped_stream_is_sent_and_logged_as_a_billed_non_stream_call(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    route: Final = respx_mock.post(_CLAIRIC_URL).respond(json=_ENVELOPE)
    recording_logger: Final = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recording_logger])

    response: Final = await litellm.aextraction(
        model="cloudflare/clairic", state=_STATE, fields=_FIELDS, stream=True, drop_params=True
    )
    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)

    assert "stream" not in json.loads(route.calls[0].request.content)
    _assert_invoice_answers(response)
    logged: Final = recording_logger.standard_logging_object
    assert logged is not None
    assert recording_logger.stream is False
    assert response.hidden_params["response_cost"] > 0
    assert logged["response_cost"] == pytest.approx(response.hidden_params["response_cost"])
