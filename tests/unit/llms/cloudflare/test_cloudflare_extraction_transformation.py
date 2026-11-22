import json
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.extraction.transformation import BaseExtractionConfig
from litellm.types.extraction import ExtractionRequestBody
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

INVOICE_STATE: Final = "Invoice #4471 from Acme Tools, dated March 3 2026. Total $174.00. Payment due in 30 days."
INVOICE_FIELDS: Final = {
    "vendor": {"type": "text", "instructions": "Who sent the invoice"},
    "invoice_date": {"type": "date"},
    "total": {"type": "number", "decimals": 2, "unit": "USD"},
    "items": {"type": "list", "items": "text"},
    "days_to_pay": {"type": "estimate", "min": 0, "max": 90, "step": 1},
}
CLAIRIC_RESULT: Final = {
    "model": "clairic",
    "answers": {
        "vendor": {
            "type": "text",
            "p_null": 0.01,
            "span": [19, 29],
            "guess": "Acme Tools",
            "confidence": 0.99,
            "value": "Acme Tools",
        },
        "total": {"type": "number", "p_null": 0.0, "guess": 174, "confidence": 0.99, "value": 174},
        "items": {"type": "list", "value": ["hammers"], "spans": [[54, 61]], "confidence": 0.4},
        "days_to_pay": {
            "type": "estimate",
            "value": 31,
            "mean": 30.7,
            "median": 30,
            "p10": 29,
            "p90": 34,
            "p_below": 0.0,
            "p_above": 0.0,
        },
    },
    "truncated": False,
    "usage": {
        "state_tokens": 52,
        "image_tokens": 0,
        "schema_tokens": 312,
        "input_tokens": 416,
        "output_tokens": 0,
        "passes": 2,
    },
}


def _cloudflare_config() -> BaseExtractionConfig:
    config: Final = ProviderConfigManager.get_provider_extraction_config("clairic", LlmProviders.CLOUDFLARE)
    assert config is not None
    return config


def _response(payload: object) -> httpx.Response:
    return httpx.Response(status_code=200, content=json.dumps(payload).encode())


def test_only_cloudflare_serves_extraction() -> None:
    assert ProviderConfigManager.get_provider_extraction_config("gpt-5", LlmProviders.OPENAI) is None
    assert isinstance(_cloudflare_config(), litellm.CloudflareExtractionConfig)


@pytest.mark.parametrize("model", ("clairic", "@cf/cloudflare/clairic"))
def test_clairic_resolves_to_its_workers_ai_run_url(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    config: Final = _cloudflare_config()
    canonical: Final = config.canonical_model(model)
    api_base: Final = config.resolve_api_base(None)

    assert canonical == "@cf/cloudflare/clairic"
    assert api_base is not None
    assert (
        config.get_complete_url(api_base, canonical)
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clairic"
    )


@pytest.mark.parametrize(
    "api_base",
    (
        "https://gateway.example/client/v4/accounts/acct/ai/v1",
        "https://gateway.example/client/v4/accounts/acct/ai/run/",
        "https://gateway.example/client/v4/accounts/acct",
    ),
)
def test_custom_api_base_keeps_the_run_path(api_base: str) -> None:
    assert (
        _cloudflare_config().get_complete_url(api_base, "@cf/cloudflare/clairic")
        == "https://gateway.example/client/v4/accounts/acct/ai/run/@cf/cloudflare/clairic"
    )


def test_missing_account_id_leaves_no_default_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)

    assert _cloudflare_config().resolve_api_base(None) is None


def test_request_forwards_the_callers_fields_unchanged() -> None:
    request: Final = ExtractionRequestBody.model_validate(
        {
            "state": INVOICE_STATE,
            "fields": INVOICE_FIELDS,
            "reference_date": "2026-03-01",
            "options": {"return_state_text": True},
            "images": [{"content_type": "image/png", "base64": "iVBORw0KGgo="}],
        }
    )

    assert _cloudflare_config().transform_extraction_request("@cf/cloudflare/clairic", request) == {
        "model": "clairic",
        "state": INVOICE_STATE,
        "fields": INVOICE_FIELDS,
        "reference_date": "2026-03-01",
        "options": {"return_state_text": True},
        "images": [{"content_type": "image/png", "base64": "iVBORw0KGgo="}],
    }


@pytest.mark.parametrize("wrapped", (True, False))
def test_response_parses_typed_answers_and_usage(wrapped: bool) -> None:
    payload: Final = (
        {"result": CLAIRIC_RESULT, "success": True, "errors": [], "messages": []} if wrapped else CLAIRIC_RESULT
    )

    response: Final = _cloudflare_config().transform_extraction_response(
        "@cf/cloudflare/clairic", "cloudflare", _response(payload)
    )

    assert response.model_dump(mode="json") == CLAIRIC_RESULT
    assert response.answers["vendor"].span == (19, 29)
    assert response.usage.input_tokens == 416


def test_unexpected_response_raises_a_provider_error() -> None:
    with pytest.raises(BaseLLMException) as error:
        _cloudflare_config().transform_extraction_response(
            "@cf/cloudflare/clairic", "cloudflare", _response({"result": {"answers": {}}})
        )

    assert error.value.status_code == 500


def test_unknown_field_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"fields\.vendor\.type"):
        ExtractionRequestBody.model_validate({"state": INVOICE_STATE, "fields": {"vendor": {"type": "color"}}})


def test_a_list_field_without_an_item_type_is_rejected_locally() -> None:
    # Cloudflare answered a list field without items with 422 code 3030 on @cf/cloudflare/clairic, probed 2026-10-20
    with pytest.raises(ValidationError, match=r"fields\.line_items\n.*list fields need items"):
        ExtractionRequestBody.model_validate({"state": INVOICE_STATE, "fields": {"line_items": {"type": "list"}}})


@pytest.mark.parametrize(
    "spec",
    (
        {"type": "number", "decimals": "2"},
        {"type": "number", "integer": "yes"},
        {"type": "number", "integer": 1},
        {"type": "estimate", "min": True, "max": 90},
    ),
    ids=("string_decimals", "string_integer", "number_integer", "bool_min"),
)
def test_a_field_value_of_the_wrong_json_type_is_rejected_instead_of_coerced(spec: dict[str, object]) -> None:
    # Cloudflare answered each of these specs with a 400 or 422 on @cf/cloudflare/clairic, probed 2026-10-20
    with pytest.raises(ValidationError, match=r"fields\.total\."):
        ExtractionRequestBody.model_validate({"state": INVOICE_STATE, "fields": {"total": spec}})


def test_numbers_in_a_field_spec_go_to_cloudflare_exactly_as_sent() -> None:
    # Cloudflare answered "decimals": 2.0 and "step": 1.0 with a 200 on @cf/cloudflare/clairic, probed 2026-10-20
    fields: Final = {
        "total": {"type": "number", "decimals": 2.0, "null_threshold": 1},
        "days_to_pay": {"type": "estimate", "min": 0, "max": 90, "step": 1.0},
    }
    request: Final = ExtractionRequestBody.model_validate({"state": INVOICE_STATE, "fields": fields})

    body: Final = _cloudflare_config().transform_extraction_request("@cf/cloudflare/clairic", request)

    assert json.dumps(body["fields"]) == json.dumps(fields)


@pytest.mark.parametrize("name", ("due date", "1st_item", "x" * 80), ids=("space", "digit_first", "80_chars"))
def test_a_field_name_clairic_refuses_is_rejected_locally(name: str) -> None:
    # Cloudflare answered each of these names with 422 code 3030 on @cf/cloudflare/clairic, probed 2026-10-19
    with pytest.raises(ValidationError, match=r"fields\."):
        ExtractionRequestBody.model_validate({"state": INVOICE_STATE, "fields": {name: {"type": "text"}}})


@pytest.mark.parametrize("name", ("_vendor", "line.items", "days-to-pay", "x" * 64))
def test_field_names_with_dots_dashes_and_underscores_up_to_64_chars_are_accepted(name: str) -> None:
    # Cloudflare answered each of these names with a 200 on @cf/cloudflare/clairic, probed 2026-10-19
    request: Final = ExtractionRequestBody.model_validate({"state": INVOICE_STATE, "fields": {name: {"type": "text"}}})

    assert tuple(request.fields) == (name,)


def test_decisions_config_still_resolves_clef_after_the_shared_helpers_moved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    config: Final = litellm.CloudflareDecisionsConfig()
    api_base: Final = config.get_default_api_base()

    assert api_base is not None
    assert (
        config.get_complete_url(api_base, config.canonical_model("clef"))
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
    )
    assert config.unwrap_response({"result": {"answers": {}}}) == {"answers": {}}


def test_enum_boolean_object_compute_and_object_list_specs_round_trip_to_clairic() -> None:
    state: Final = "Order #A17: Maria bought 3 red mugs at $4.50 each and paid by credit card. Gift wrap was requested."
    fields: Final = {
        "payment": {
            "type": "enum",
            "enum": ["cash", "credit card", "paypal"],
            "instructions": "Payment method",
        },
        "gift_wrap": {"type": "boolean", "instructions": "Whether gift wrap was requested"},
        "item": {
            "type": "object",
            "instructions": "The purchased item",
            "properties": {
                "name": {"type": "text"},
                "qty": {"type": "number"},
                "gift": {"type": "boolean"},
                "color": {"type": "enum", "enum": ["red", "blue"]},
            },
        },
        "total": {"type": "compute", "instructions": "Total cost in dollars", "unit": "USD"},
        "rows": {
            "type": "list",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "text"}, "qty": {"type": "number"}},
            },
            "instructions": "Items purchased",
        },
    }
    request: Final = ExtractionRequestBody.model_validate({"state": state, "fields": fields})

    body: Final = _cloudflare_config().transform_extraction_request("@cf/cloudflare/clairic", request)

    assert json.loads(json.dumps(body)) == {"model": "clairic", "state": state, "fields": fields}


def test_response_parses_enum_boolean_object_compute_and_object_list_answers() -> None:
    result: Final = {
        "model": "clairic-27b",
        "answers": {
            "payment": {"type": "enum", "value": "credit card", "raw": '"credit card"'},
            "gift_wrap": {"type": "boolean", "value": True},
            "item": {
                "type": "object",
                "value": {"name": "mugs", "qty": 3, "gift": True, "color": "red"},
                "raw": '{"name":"mugs","qty":3,"gift":true,"color":"red"}',
            },
            "total": {"type": "compute", "value": 16.5, "guess": 16.5},
            "rows": {
                "type": "list",
                "value": [{"name": "red mugs", "qty": 3}],
                "property_spans": [{"name": [27, 35], "qty": [25, 26]}],
            },
        },
        "truncated": False,
        "usage": {"input_tokens": 220, "output_tokens": 36},
    }

    response: Final = _cloudflare_config().transform_extraction_response(
        "@cf/cloudflare/clairic",
        "cloudflare",
        _response({"result": result, "success": True, "errors": [], "messages": []}),
    )

    assert response.model_dump(mode="json") == result
    assert response.answers["payment"].value == "credit card"
    assert response.answers["gift_wrap"].value is True
    assert response.answers["item"].value == {"name": "mugs", "qty": 3, "gift": True, "color": "red"}
    assert response.answers["total"].value == 16.5
    assert response.answers["rows"].property_spans == ({"name": [27, 35], "qty": [25, 26]},)


@pytest.mark.parametrize(
    ("field_spec", "pattern"),
    (
        ({"type": "enum"}, r"enum fields need enum"),
        ({"type": "enum", "enum": []}, r"fields\.target\.enum"),
        ({"type": "enum", "enum": "cash"}, r"fields\.target\.enum"),
        ({"type": "object"}, r"object fields need properties"),
        ({"type": "object", "properties": {}}, r"fields\.target\.properties"),
        ({"type": "object", "properties": {"bad name": {"type": "text"}}}, r"fields\.target\.properties"),
        ({"type": "object", "properties": {"nested": {"type": "object"}}}, r"fields\.target\.properties\.nested"),
        ({"type": "object", "properties": {"color": {"type": "enum"}}}, r"enum properties need enum"),
        ({"type": "list", "items": {"type": "object"}}, r"fields\.target\.items"),
        ({"type": "list", "items": {"type": "object", "properties": {}}}, r"fields\.target\.items"),
        ({"type": "list", "items": "boolean"}, r"fields\.target\.items"),
    ),
    ids=(
        "enum_missing_options",
        "enum_empty_options",
        "enum_non_array",
        "object_missing_properties",
        "object_empty_properties",
        "object_invalid_property_name",
        "object_unsupported_property_type",
        "object_enum_property_missing_options",
        "list_object_missing_properties",
        "list_object_empty_properties",
        "list_unsupported_scalar_item",
    ),
)
def test_invalid_enum_object_and_list_item_specs_are_rejected_locally(
    field_spec: dict[str, object], pattern: str
) -> None:
    with pytest.raises(ValidationError, match=pattern):
        ExtractionRequestBody.model_validate({"state": INVOICE_STATE, "fields": {"target": field_spec}})
